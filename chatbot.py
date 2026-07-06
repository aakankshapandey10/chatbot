import os
import sys

import anthropic
from dotenv import load_dotenv
from anthropic import Anthropic, APIError, APIConnectionError, AuthenticationError

MODEL = "claude-haiku-4-5"


def main():
    # --- setup ---
    load_dotenv()  # injects key/value pairs from .env into os.environ

    if not os.environ.get("AZURE_API_KEY"):
        # fail fast with a readable message instead of a cryptic SDK auth error later
        print("Error: AZURE_API_KEY not found. Make sure it is set in your .env file.")
        sys.exit(1)

    # client bound to Azure's endpoint instead of the default Anthropic API
    client = Anthropic(base_url=os.getenv("AZURE_BASE_URL"), api_key=os.getenv("AZURE_API_KEY"))
    history = []  # mutable conversation memory: list of {"role", "content"} dicts

    print("Claude command-line chatbot. Type 'exit' or 'quit' to leave.")

    # --- loop control: runs forever until a break below ---
    while True:
        # --- input handler ---
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # stdin closed (Ctrl-D) or Ctrl-C: treat as "user wants to leave"
            print("\nGoodbye!")
            break

        if not user_input:
            # blank line: don't waste an API call, just re-prompt
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # --- history/state management: record the user's turn before calling the API ---
        history.append({"role": "user", "content": user_input})

        # --- API call wrapper ---
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=history,  # full conversation so far, not just the latest message
            )
        except AuthenticationError:
            print("Error: authentication failed. Check that your API key is valid.")
            history.pop()  # roll back the orphaned user turn so history stays valid
            continue
        except APIConnectionError:
            print("Error: could not connect to the Anthropic API. Check your network connection.")
            history.pop()
            continue
        except APIError as e:
            print(f"Error: the API returned an error: {e}")
            history.pop()
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            history.pop()
            continue

        # --- reply extraction and output ---
        # response.content can hold multiple block types; keep only text blocks
        reply_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        print(reply_text)

        # record the assistant's turn so the next call includes it as context
        history.append({"role": "assistant", "content": reply_text})


# --- entry point: only run main() when executed directly, not on import ---
if __name__ == "__main__":
    main()
