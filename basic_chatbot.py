"""
LangGraph + Ollama + SearXNG (Raspberry Pi) + Human-in-the-Loop
- Local LLM via Ollama (e.g., `mistral-nemo:12b`)
- Web search tool via SearXNG JSON API (hosted on your Pi)
- Memory via LangGraph checkpointer
- Human-in-the-loop using interrupt()/Command()
"""

from __future__ import annotations

# ---------- Stdlib ----------
import json
import os
from typing import Annotated, List, Dict, Any

# ---------- Third-party ----------
import httpx
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langgraph.errors import GraphInterrupt  # allow interrupts to bubble

# ========== Configuration ==========
SEARXNG_HOST: str = os.getenv("SEARXNG_HOST", "http://192.168.68.63")
SEARXNG_TIMEOUT_S: float = float(os.getenv("SEARXNG_TIMEOUT_S", "30"))
DEFAULT_RESULTS: int = int(os.getenv("SEARXNG_RESULTS", "5"))
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")
THREAD_ID: str = os.getenv("THREAD_ID", "cli-default")

# ========== State definition ==========
class State(TypedDict):
    # Annotated + add_messages => LangGraph appends (not overwrite) each step
    messages: Annotated[list, add_messages]

# ========== LLM (local, via Ollama) ==========
llm = ChatOllama(model=OLLAMA_MODEL)

# ========== Tools ==========
# ---- SearXNG Web Search ----
def _searxng_search(query: str, num_results: int) -> Dict[str, Any]:
    """Call SearXNG JSON API running on the Pi."""
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

# ---- Human-in-the-loop tool ----
@tool("human_assistance")
def human_assistance(query: str) -> str:
    """
    Request assistance from a human. Pauses the graph using interrupt().
    On resume, returns the human's text.
    """
    # You can surface any structured payload you want.
    # The docs commonly show a dict with 'query' and expect resume={"data": "..."}.
    prompt_payload = {"query": query}
    human_response = interrupt(prompt_payload)  # pauses; resumes with Command(resume=...)
    # Expect Command(resume={"data": "<human text>"})
    if isinstance(human_response, dict) and "data" in human_response:
        return human_response["data"]
    # Fallback: if someone resumed with a plain string
    return str(human_response)

TOOLS = [web_search, human_assistance]

# Bind tool schemas so the model can emit tool_calls
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Graph nodes ==========
def chatbot(state: State) -> Dict[str, List]:
    """Call the LLM; it may emit a tool_call."""
    reply = llm_with_tools.invoke(state["messages"])
    # HITL safety: avoid parallel tool calls since resuming could duplicate work
    if getattr(reply, "tool_calls", None):
        assert len(reply.tool_calls) <= 1, "Parallel tool calls disabled for HITL safety."
    return {"messages": [reply]}

class BasicToolNode:
    """
    Execute tool calls emitted by the last AI message.
    Adds a ToolMessage for each executed tool call.
    Lets GraphInterrupt bubble so the runtime can pause properly.
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
                out_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        name=tool_name or "unknown",
                        tool_call_id=tc.get("id", ""),
                    )
                )
                continue

            try:
                # IMPORTANT: do NOT swallow GraphInterrupt
                result = tool_obj.invoke(args)
                payload = json.dumps(result, ensure_ascii=False)
            except GraphInterrupt:
                # Re-raise so LangGraph can pause execution and surface __interrupt__.
                raise
            except Exception as e:
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

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Memory / persistence is REQUIRED for interrupts
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# ========== Runner with interrupt handling ==========
def _print_ai_from_event(evt: Dict[str, Any]) -> None:
    """Pretty-print the latest assistant content if present."""
    if "messages" in evt and evt["messages"]:
        last = evt["messages"][-1]
        # 'content' may be list or str; keep it simple for CLI
        content = getattr(last, "content", None)
        if content:
            print("Assistant:", content)

def _handle_interrupts_and_resume(config: Dict[str, Any]) -> bool:
    """
    Check if the graph is paused (interrupts present) and, if so,
    ask user for input and resume. Returns True if we resumed, else False.
    """
    # When streaming with StreamMode="values", interrupts appear as "__interrupt__"
    # (see LangGraph types reference). :contentReference[oaicite:1]{index=1}
    # However, we can also just inspect get_state and ask.
    snapshot = graph.get_state(config)
    if not snapshot.interrupts:
        return False

    # For simplicity we handle the next (first) pending interrupt.
    intr = snapshot.interrupts[0]
    payload = intr.value  # what we surfaced in interrupt(...)
    print("\n--- HUMAN NEEDED ---")
    if isinstance(payload, dict) and "query" in payload:
        print("Request:", payload["query"])
        human_text = input("Your reply: ")
        resume_value = {"data": human_text}
    else:
        print("Request:", payload)
        resume_value = input("Your reply: ")

    # Resume from the same node using Command(resume=...)
    for event in graph.stream(Command(resume=resume_value), config, stream_mode="values"):
        if "__interrupt__" in event:
            # If we hit another interrupt immediately (nested), recurse-like behavior:
            _print_ai_from_event(event)
            return _handle_interrupts_and_resume(config)
        _print_ai_from_event(event)
    return True

def stream_graph_updates(user_input: str, thread_id: str) -> None:
    """Send one user message; stream assistant updates, handle interrupts if any."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [{"role": "user", "content": user_input}]}

    # First pass: run until completion OR until we hit an interrupt
    for event in graph.stream(initial_state, config, stream_mode="values"):
        if "__interrupt__" in event:
            # Pause & collect human input, then resume
            _print_ai_from_event(event)
            _handle_interrupts_and_resume(config)
            # After resuming, we keep going only if further interrupts happen.
            break
        _print_ai_from_event(event)

if __name__ == "__main__":
    print(f"[info] Using SearXNG at: {SEARXNG_HOST}")
    print(f"[info] Using Ollama model: {OLLAMA_MODEL}")
    print(f"[info] Thread ID: {THREAD_ID}")
    print("[info] Type 'quit' to exit.\n")
    while True:
        q = input("User: ")
        if q.strip().lower() in {"q", "quit", "exit"}:
            break
        stream_graph_updates(q, THREAD_ID)
