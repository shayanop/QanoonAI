"""Model backends. Do not edit.

- hf      : the official backend. Loads the pinned base model with Hugging Face
            transformers (4-bit on a CUDA GPU) and, optionally, your LoRA adapter.
            All submitted results must come from this backend.
- openai  : any OpenAI-compatible server (vLLM, Ollama, LM Studio) serving the SAME
            pinned model. Handy for fast development. Set QA_OPENAI_BASE_URL,
            QA_OPENAI_MODEL and, if needed, QA_OPENAI_API_KEY.
- mock    : no model at all; returns a fixed-rule answer. Only for checking that
            your pipeline runs (CI, quick tests). Never submit mock results.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .common import approx_tokens, load_config


class Backend:
    name = "base"

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)

    def generate(self, messages: list[dict], seed: int) -> tuple[str, int, int]:
        raise NotImplementedError


class MockBackend(Backend):
    name = "mock"

    def generate(self, messages, seed):
        prompt = "\n".join(m["content"] for m in messages)
        docs = prompt.split("DOCUMENTS:", 1)[-1] if "DOCUMENTS:" in prompt else ""
        first = re.search(r"^\[([^\]\s]+)\]\s*(.*)$", docs, flags=re.M)
        if not first:
            out = {"answer": "I am not sure; the documents do not contain the answer.", "citations": [], "abstain": True}
        else:
            doc_id = first.group(1).split("#")[0]
            out = {"answer": f"The relevant provision is {first.group(2)[:200]}", "citations": [doc_id], "abstain": False}
        text = json.dumps(out)
        return text, approx_tokens(prompt), approx_tokens(text)


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = os.environ.get("QA_OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/") + "/chat/completions"
        self.model = os.environ.get("QA_OPENAI_MODEL", cfg["model"]["base_model"])
        self.key = os.environ.get("QA_OPENAI_API_KEY", "none")

    def generate(self, messages, seed):
        d = self.cfg["decoding"]
        body = {
            "model": self.model, "messages": messages, "temperature": d["temperature"],
            "top_p": d["top_p"], "max_tokens": d["max_new_tokens"], "seed": seed,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        prompt = "\n".join(m["content"] for m in messages)
        return text, usage.get("prompt_tokens", approx_tokens(prompt)), usage.get("completion_tokens", approx_tokens(text))


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.cfg = torch, cfg
        base = cfg["model"]["base_model"]
        self.tok = AutoTokenizer.from_pretrained(base)
        kwargs = {}
        if torch.cuda.is_available():
            if cfg["model"].get("load_in_4bit", True):
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                )
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages, seed):
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        d = self.cfg["decoding"]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=d["max_new_tokens"], do_sample=d["temperature"] > 0,
                temperature=d["temperature"], top_p=d["top_p"], pad_token_id=self.tok.eos_token_id,
            )
        new = out[0][ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), int(ids.shape[1]), int(new.shape[0])


def make_backend(name: str | None = None, adapter: str | None = None, cfg: dict | None = None) -> Backend:
    cfg = cfg or load_config()
    name = (name or os.environ.get("QA_BACKEND") or "hf").lower()
    if name == "mock":
        return MockBackend()
    if name == "openai":
        if adapter:
            print("[backend] note: with the openai backend the server must already serve your adapter.")
        return OpenAICompatBackend(cfg)
    if name == "hf":
        return HFBackend(cfg, adapter)
    raise ValueError(f"Unknown backend '{name}'. Use hf, openai or mock.")
