"""
LangGraph + Ollama + SearXNG (hosted on Raspberry Pi)
- Local LLM via Ollama (e.g., `mistral-nemo:12b`)
- Web search tool via SearXNG JSON API
- Tool-calling loop with LangGraph
- Persistent chat memory via LangGraph checkpointer (InMemorySaver)

Run:
  python add_tools_searxng_pi.py
"""

from __future__ import annotations

# ---------- Standard library ----------
import json
import os
from typing import Annotated, List, Dict, Any

# ---------- Third-party ----------
from typing_extensions import TypedDict
import httpx

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver  # <-- memory checkpointer

# ========== Configuration ==========
# Set your Pi host once; allow override via env var.
# Tip: Include the port (e.g., 8080) if your SearXNG runs there.
SEARXNG_HOST: str = os.getenv("SEARXNG_HOST", "http://192.168.68.63")
SEARXNG_TIMEOUT_S: float = float(os.getenv("SEARXNG_TIMEOUT_S", "30"))
DEFAULT_RESULTS: int = int(os.getenv("SEARXNG_RESULTS", "5"))
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")

# Memory / session
THREAD_ID: str = os.getenv("THREAD_ID", "1")  # choose a different value to start a fresh thread

# ========== State definition ==========
class State(TypedDict):
    # Use Annotated + add_messages so LangGraph appends instead of overwriting.
    messages: Annotated[list, add_messages]

# ========== LLM (local, via Ollama) ==========
llm = ChatOllama(model=OLLAMA_MODEL)

# ========== Tool: SearXNG Web Search ==========
def _searxng_search(query: str, num_results: int) -> Dict[str, Any]:
    """Internal helper that calls SearXNG with JSON output enabled."""
    params = {"q": query, "format": "json", "pageno": 1}
    with httpx.Client(timeout=SEARXNG_TIMEOUT_S) as client:
        resp = client.get(f"{SEARXNG_HOST}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
            }
        )
    return {"results": results}

@tool("web_search")
def web_search(query: str, num_results: int = DEFAULT_RESULTS) -> Dict[str, Any]:
    """
    Search the web via your Raspberry Pi SearXNG instance.
    Returns: {"results": [{"title": str, "url": str, "snippet": str}, ...]}
    """
    if not query or not isinstance(query, str):
        return {"results": [], "error": "Empty or invalid query"}
    try:
        return _searxng_search(query, num_results)
    except httpx.HTTPStatusError as e:
        return {"results": [], "error": f"SearXNG HTTP error: {e.response.status_code}"}
    except httpx.RequestError as e:
        return {"results": [], "error": f"SearXNG connection error: {e}"}

TOOLS = [web_search]

# LLM with tool “schema” bound so it knows how to call them.
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Graph nodes ==========
def chatbot(state: State) -> Dict[str, List]:
    """Calls the LLM with the running message list; may emit tool_calls."""
    reply = llm_with_tools.invoke(state["messages"])
    return {"messages": [reply]}

class BasicToolNode:
    """
    Executes tool calls emitted by the last AI message.
    Adds a ToolMessage response for each executed tool call.
    """
    def __init__(self, tools: List):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")

        last = messages[-1]
        out_messages: List[ToolMessage] = []

        for tc in getattr(last, "tool_calls", []) or []:
            tool_name = tc.get("name")
            args = tc.get("args", {})

            tool_obj = self.tools_by_name.get(tool_name)
            if tool_obj is None:
                # Return a tool error back to the model; it may recover.
                out_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        name=tool_name or "unknown",
                        tool_call_id=tc.get("id", ""),
                    )
                )
                continue

            try:
                result = tool_obj.invoke(args)
                payload = json.dumps(result, ensure_ascii=False)
            except Exception as e:  # keep going; let the LLM handle failures
                payload = json.dumps({"error": f"Tool failed: {e}"}, ensure_ascii=False)

            out_messages.append(
                ToolMessage(
                    content=payload,
                    name=tool_name,
                    tool_call_id=tc.get("id", ""),
                )
            )

        return {"messages": out_messages}

# ========== Graph wiring ==========
graph_builder = StateGraph(State)

tool_node = BasicToolNode(TOOLS)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

def route_tools(state: State):
    """If the last AI message asked for tools, go to 'tools', else END."""
    msgs = state.get("messages", [])
    if not msgs:
        return END
    ai_msg = msgs[-1]
    if getattr(ai_msg, "tool_calls", None):
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# After tools execute, loop back to the chatbot for finalization/next step.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# ----- Add memory (checkpointer) per tutorial -----
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)  # <-- enable persistence
# (In production, consider SqliteSaver/PostgresSaver; this in-memory saver resets when the process restarts.)

# ========== Runner ==========
def stream_graph_updates(user_input: str, thread_id: str) -> None:
    """
    Send a single user message and stream assistant updates as they occur.
    Memory is keyed by `thread_id` via the config's configurable block.
    """
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    config = {"configurable": {"thread_id": thread_id}}  # <-- critical for memory
    # Using stream_mode="values" to yield only the state values (simpler to print)
    for event in graph.stream(initial_state, config, stream_mode="values"):
        # Each `event` is the current state values snapshot
        print("Assistant:", event["messages"][-1].content)

if __name__ == "__main__":
    print(f"[info] Using SearXNG at: {SEARXNG_HOST}")
    print(f"[info] Using Ollama model: {OLLAMA_MODEL}")
    print(f"[info] Using THREAD_ID: {THREAD_ID}  (set env THREAD_ID to change session)")
    print("[help] Type 'new:<id>' to switch to a new thread at any time (e.g., new:2).")
    thread_id = THREAD_ID
    while True:
        q = input("User: ")
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q.lower().startswith("new:"):
            thread_id = q.split(":", 1)[1].strip() or thread_id
            print(f"[info] Switched to THREAD_ID={thread_id}")
            continue
        stream_graph_updates(q, thread_id)
