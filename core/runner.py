# === harin/core/runner.py ===
# HarinAgent runner entry point

from harin.core.harin_main_loop import run_harin_session

def harin_respond(user_input: str) -> dict:
    """
    Public API to run the full HarinAgent v6.1 reasoning loop.

    Returns a dict with:
    {
        "output": Final text output (possibly corrected),
        "score": Composite judgment score,
        "verifier": Self-verification summary,
        "trace": List of step summaries
    }
    """
    return run_harin_session(user_input)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("🧠 harin > ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            result = harin_respond(user_input)
            print("\n🟢 OUTPUT:", result["output"])
            print("🧪 SCORE:", result["score"])
            print("🧭 TRACE:")
            for step in result["trace"]:
                print("   •", step)
            print()
        except KeyboardInterrupt:
            break