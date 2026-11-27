"""Memory factory: get_memory("M0" | "M1" | "M2" | "M3")."""
from .base import Memory, MemoryRead, fit_to_budget
from .m0_window import WindowMemory
from .m1_summary import SummaryMemory
from .m2_facts import FactMemory
from .m3_temporal import TemporalGraphMemory

_REGISTRY = {"M0": WindowMemory, "M1": SummaryMemory, "M2": FactMemory, "M3": TemporalGraphMemory}


def get_memory(code: str) -> Memory:
    code = code.upper()
    if code not in _REGISTRY:
        raise ValueError(f"Unknown memory '{code}'. Use one of {sorted(_REGISTRY)}.")
    return _REGISTRY[code]()


__all__ = ["Memory", "MemoryRead", "fit_to_budget", "get_memory"]
