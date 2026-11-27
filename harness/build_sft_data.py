"""Build fine-tuning data in the exact prompt format the chatbot uses (Step 3). Do not edit.

    python -m harness.build_sft_data --out data_sft.jsonl

For every gold turn in the training split it builds the same prompt the agent sends at
run time (fixed system prompt, retrieved documents, the earlier turns as MEMORY with the
M0 window) and the target reply in the required JSON answer format. The output has one
{"messages": [...]} chat per line, ready for TRL's SFTTrainer.

It also writes <out>.dpo_seed.jsonl with {"prompt", "chosen"} pairs. A2 groups build the
"rejected" side themselves (see the notebook) and must document how.
"""
from __future__ import annotations

import argparse
import json

from harness.answer import Answer
from harness.backends import MockBackend

from .common import ROOT, Corpus, load_config, read_jsonl, write_jsonl
from .retriever import BM25Retriever


def target_json(gold: dict) -> str:
    return json.dumps({"answer": gold["answer"], "citations": gold.get("citations", []),
                       "abstain": bool(gold.get("abstain"))}, ensure_ascii=False)


def build(split_file, limit: int):
    from agent.graph import LegalAgent
    from agent.memory import WindowMemory

    cfg = load_config()
    corpus = Corpus()
    agent = LegalAgent(WindowMemory(), MockBackend(), cfg, corpus, BM25Retriever(corpus, cfg))
    sft, dpo = [], []
    for sess in read_jsonl(split_file):
        sid = sess["session_id"]
        agent.reset(sid)
        for ti, turn in enumerate(sess["turns"], 1):
            state = {"session_id": sid, "turn": ti, "user_msg": turn["user"]}
            state.update(agent.memory_read(state))
            state.update(agent.retrieve(state))
            messages = agent._build_messages(state)
            gold = turn.get("gold")
            reply = target_json(gold) if gold else json.dumps(
                {"answer": "Noted.", "citations": [], "abstain": False})
            if gold:
                sft.append({"messages": messages + [{"role": "assistant", "content": reply}],
                            "session_id": sid, "turn": ti, "type": sess.get("type")})
                dpo.append({"prompt": messages, "chosen": [{"role": "assistant", "content": reply}],
                            "gold_citations": gold.get("citations", []), "abstain": bool(gold.get("abstain")),
                            "session_id": sid, "turn": ti, "type": sess.get("type")})
            # memory sees the reference answer, as it would see the bot's answer at run time
            g = gold or {"answer": "Noted.", "citations": []}
            agent.memory.write(sid, ti, turn["user"], Answer(g["answer"], g.get("citations", [])), state)
            if limit and len(sft) >= limit:
                return sft, dpo
    return sft, dpo


def main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description="Build SFT / DPO seed data from the training split")
    p.add_argument("--out", default="data_sft.jsonl")
    p.add_argument("--split-file", default=str(ROOT / cfg["paths"]["data_dir"] / "train.jsonl"))
    p.add_argument("--limit", type=int, default=cfg["training"]["max_train_examples"])
    a = p.parse_args(argv)
    sft, dpo = build(a.split_file, a.limit)
    write_jsonl(a.out, sft)
    write_jsonl(a.out.replace(".jsonl", "") + ".dpo_seed.jsonl", dpo)
    print(f"wrote {len(sft)} SFT examples to {a.out} (limit {a.limit}) and DPO seeds alongside")


if __name__ == "__main__":
    main()
