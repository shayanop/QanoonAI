"""Shared loaders and helpers for the QanoonArena harness. Do not edit."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("T1", "T2", "T3", "T4", "T5")


# ---------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "harness" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return sha256_text(json.dumps(cfg, sort_keys=True))[:12]


# ---------------------------------------------------------------- text
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_text(text: str) -> str:
    """Lower-case, collapse spaces, unify dashes and quotes. Used for matching."""
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def question_hash(text: str) -> str:
    return sha256_text(norm_text(text))


def approx_tokens(text: str) -> int:
    """Fallback token count (about 4 characters per token). Backends override this."""
    return max(1, len(text) // 4) if text else 0


def file_hash(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    p = Path(path)
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    for fp in files:
        h.update(fp.name.encode())
        with open(fp, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- jsonl
def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {n} is not valid JSON ({e})") from e
    return rows


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- corpus
@dataclass
class Doc:
    doc_id: str                 # e.g. "PPC-378" or "PPC-378@2017-03-01" or "SHC-2019-0123"
    kind: str                   # "statute" or "judgment"
    title: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def base_id(self) -> str:
        """Version-free id: PPC-378@2017-03-01 -> PPC-378."""
        return self.doc_id.split("@")[0]


class Corpus:
    def __init__(self, corpus_dir: str | os.PathLike | None = None):
        cfg_dir = corpus_dir or ROOT / load_config()["paths"]["corpus_dir"]
        self.dir = Path(cfg_dir)
        self.docs: dict[str, Doc] = {}
        for name, kind in (("statutes.jsonl", "statute"), ("judgments.jsonl", "judgment")):
            fp = self.dir / name
            if not fp.exists():
                continue
            for r in read_jsonl(fp):
                meta = {k: v for k, v in r.items() if k not in ("doc_id", "title", "text")}
                self.docs[r["doc_id"]] = Doc(r["doc_id"], kind, r.get("title", ""), r["text"], meta)
        if not self.docs:
            raise FileNotFoundError(
                f"No corpus found in {self.dir}. The instructor's prepare step fills this folder."
            )

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, doc_id: str) -> Doc | None:
        return self.docs.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        """True for an exact id, or a version-free id that has at least one version."""
        if doc_id in self.docs:
            return True
        return any(d.base_id == doc_id for d in self.docs.values())

    def version_in_force(self, base_id: str, as_of: str) -> Doc | None:
        """The statute version valid on date as_of (YYYY-MM-DD), or None."""
        best = None
        for d in self.docs.values():
            if d.base_id != base_id:
                continue
            vf = d.meta.get("valid_from") or "0000-00-00"
            vt = d.meta.get("valid_to") or "9999-12-31"
            if vf <= as_of < vt or (vf <= as_of and vt == "9999-12-31"):
                best = d
        return best


def same_citation(pred: str, gold: str, exact_version: bool) -> bool:
    """A predicted citation matches gold. Versions matter only when exact_version is true."""
    pred, gold = pred.strip(), gold.strip()
    if exact_version:
        return pred == gold
    return pred.split("@")[0] == gold.split("@")[0]
