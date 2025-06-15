# === harin/interface/live_runner.py ===
# Streamlit / CLI / REST compatible live interface for HarinAgent

from harin.core.runner import harin_respond

def interactive_loop():
    print("🧠 HarinAgent v6.1 – Interactive Mode")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you > ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            result = harin_respond(user_input)
            print("\n🔹 Harin:", result["output"])
            print("🧪 score:", result["score"])
            print("📋 trace:", " → ".join(result["trace"]))
            print()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")