"""QanoonArena agent package.

    from agent import build_agent
    bot = build_agent(memory="M0", backend="hf", adapter_path=None)
    bot.reset("demo")
    answer, info = bot.respond("demo", 1, "Which section of the PPC defines theft?")
    print(answer.display())
"""
from harness.backends import make_backend
from harness.common import load_config

from .graph import LegalAgent
from .memory import get_memory


def build_agent(memory: str = "M0", backend: str | None = None, adapter_path: str | None = None,
                seed: int = 13, tracer=None, cfg: dict | None = None) -> LegalAgent:
    cfg = cfg or load_config()
    be = make_backend(backend, adapter_path, cfg)
    return LegalAgent(get_memory(memory), be, cfg=cfg, tracer=tracer, seed=seed)


__all__ = ["build_agent", "LegalAgent"]
