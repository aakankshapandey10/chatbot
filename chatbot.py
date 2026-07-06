import os
import sys

import anthropic
from ddgs import DDGS
from dotenv import load_dotenv
from anthropic import Anthropic, APIError, APIConnectionError, AuthenticationError

MODEL = "claude-haiku-4-5"

COMPACT_THRESHOLD = 10  # once history exceeds this many messages, fold old ones into a summary
KEEP_RECENT = 4  # always leave this many messages untouched, verbatim

# fix identity (models are unreliable at self-reporting this); knowledge cutoff is no longer
# hedged about since the model can now look things up with web_search instead of guessing.
BASE_SYSTEM_PROMPT = (
    "You are Claude Haiku 4.5, made by Anthropic. For anything time-sensitive (current "
    "officeholders, recent events, prices, versions) or anything you are not confident is "
    "still accurate, use the web_search tool rather than relying on your training data."
)

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Use for anything time-sensitive or recent "
        "that your training data may not cover accurately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
}


def web_search(query, max_results=5):
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}" for r in results
    )


def _content_to_text(content):
    # message content is a plain string for normal turns, but a list of blocks
    # (tool_use / tool_result / SDK block objects) for tool-use turns
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "text":
            parts.append(block["text"] if isinstance(block, dict) else block.text)
        elif block_type == "tool_use":
            query = block.get("input", {}).get("query") if isinstance(block, dict) else block.input.get("query")
            parts.append(f"[searched for: {query}]")
        elif block_type == "tool_result":
            parts.append(f"[search result: {block.get('content', '')}]")
    return " ".join(parts)


def summarize_turns(old_messages, client):
    # separate, throwaway API call: does not read or write the live `history`
    summary_prompt = [{
        "role": "user",
        "content": (
            "Summarize the following conversation concisely, preserving key facts, "
            "names, decisions, and anything the user would expect to be remembered:\n\n"
            + "\n".join(f"{m['role']}: {_content_to_text(m['content'])}" for m in old_messages)
        ),
    }]
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=summary_prompt,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def compact_history(history, client, existing_summary):
    # split off everything except the most recent turns, summarize just that chunk,
    # and fold it into the running summary instead of replacing it (so earlier
    # compactions aren't lost when a later one happens)
    split = len(history) - KEEP_RECENT
    # don't split in the middle of a tool_use/tool_result exchange — the API requires a
    # tool_use to be immediately followed by its matching tool_result. Walk the split point
    # back until "recent" starts on a plain-text message instead of a block list.
    while split > 0 and not isinstance(history[split]["content"], str):
        split -= 1
    old, recent = history[:split], history[split:]
    new_chunk_summary = summarize_turns(old, client)
    combined_summary = f"{existing_summary} {new_chunk_summary}" if existing_summary else new_chunk_summary
    return recent, combined_summary


def get_assistant_reply(client, history, system_prompt):
    # loop until Claude returns a final text answer instead of requesting a tool call;
    # appends every intermediate assistant/tool-result turn to `history` as it goes
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=history,
            system=system_prompt,
            tools=[WEB_SEARCH_TOOL],
        )
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = [
            {"type": "tool_result", "tool_use_id": block.id, "content": web_search(block.input["query"])}
            for block in response.content
            if block.type == "tool_use" and block.name == "web_search"
        ]
        history.append({"role": "user", "content": tool_results})


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
    summary = None  # running system-message summary of turns folded out of `history`

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
        rollback_point = len(history) - 1  # discard this whole turn (incl. any tool round-trips) on failure

        system_prompt = BASE_SYSTEM_PROMPT
        if summary:
            system_prompt += f"\n\nSummary of earlier conversation: {summary}"

        # --- API call wrapper (may loop internally for tool_use round-trips) ---
        try:
            reply_text = get_assistant_reply(client, history, system_prompt)
        except AuthenticationError:
            print("Error: authentication failed. Check that your API key is valid.")
            del history[rollback_point:]  # roll back the orphaned turn so history stays valid
            continue
        except APIConnectionError:
            print("Error: could not connect to the Anthropic API. Check your network connection.")
            del history[rollback_point:]
            continue
        except APIError as e:
            print(f"Error: the API returned an error: {e}")
            del history[rollback_point:]
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            del history[rollback_point:]
            continue

        print(reply_text)

        # --- compaction: keep the live message list from growing unbounded ---
        if len(history) > COMPACT_THRESHOLD:
            history, summary = compact_history(history, client, summary)


# --- entry point: only run main() when executed directly, not on import ---
if __name__ == "__main__":
    main()
