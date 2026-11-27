"""Talk to your chatbot in the terminal.

    python scripts/chat.py --memory M0
    python scripts/chat.py --memory M2 --adapter-path ./adapters/a1 --backend hf

Type a message and press Enter. Type 'reset' for a new conversation, 'quit' to exit.
Traces of this chat go to traces/chat_<time>.jsonl.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_agent  # noqa: E402
from harness.tracing import Tracer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", default="M0")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--backend", default=None)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    tracer = Tracer(f"chat_{time.strftime('%Y%m%d-%H%M%S')}", "traces")
    bot = build_agent(a.memory, a.backend, a.adapter_path, a.seed, tracer)
    sid, turn = "chat-1", 0
    bot.reset(sid)
    print("QanoonArena chat. Type 'reset' for a new conversation, 'quit' to exit.\n")
    try:
        while True:
            msg = input("You: ").strip()
            if not msg:
                continue
            if msg.lower() == "quit":
                break
            if msg.lower() == "reset":
                sid, turn = f"chat-{int(time.time())}", 0
                bot.reset(sid)
                print("(new conversation)\n")
                continue
            turn += 1
            ans, info = bot.respond(sid, turn, msg)
            print(f"\nBot: {ans.display()}\n   [{info['tokens_in']}+{info['tokens_out']} tokens, {info['latency_ms']} ms]\n")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        tracer.close()


if __name__ == "__main__":
    main()
