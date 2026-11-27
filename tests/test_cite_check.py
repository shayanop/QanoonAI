import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.cite_check import check
from harness.answer import Answer


def test_cite_check():
    allowed = {"PPC-302", "CPC-9"}

    clean_ans = Answer(answer="Murder is defined under PPC-302.", citations=["PPC-302"], abstain=False)
    _, clean_flags = check(clean_ans, allowed, corpus=None)
    assert clean_flags["n_ungrounded"] == 0, f"Expected 0 ungrounded, got {clean_flags}"
    assert clean_flags["ungrounded"] == []

    dirty_ans = Answer(answer="Test answer.", citations=["PPC-302", "FAKE-999"], abstain=False)
    _, dirty_flags = check(dirty_ans, allowed, corpus=None)
    assert dirty_flags["n_ungrounded"] == 1, f"Expected 1 ungrounded, got {dirty_flags}"
    assert dirty_flags["ungrounded"] == ["FAKE-999"]

    print("test_cite_check PASSED")


if __name__ == "__main__":
    test_cite_check()
