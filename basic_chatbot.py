"""
LangGraph + Ollama + SearXNG (hosted on Raspberry Pi)
- Local LLM via Ollama (e.g., `mistral-nemo:12b`)
- Web search tool via SearXNG JSON API
- Tool-calling loop with LangGraph
- Persistent chat memory via LangGraph checkpointer (InMemorySaver)
- Human-in-the-loop via `interrupt` + `Command(resume=...)`
- Customized state: adds 'name' and 'birthday' fields and updates them inside a tool
"""

from __future__ import annotations

# ---------- Standard library ----------
import json
import os
from typing import Annotated, List, Dict, Any, Optional

# ---------- Third-party ----------
from typing_extensions import TypedDict
import httpx

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt


# ========== Configuration ==========
SEARXNG_HOST: str = os.getenv("SEARXNG_HOST", "http://192.168.68.63")
SEARXNG_TIMEOUT_S: float = float(os.getenv("SEARXNG_TIMEOUT_S", "30"))
DEFAULT_RESULTS: int = int(os.getenv("SEARXNG_RESULTS", "5"))
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b")

# Memory / session
THREAD_ID: str = os.getenv("THREAD_ID", "1")  # choose a different value to start a fresh thread

# ========== State definition ==========
class State(TypedDict):
    # Append new messages instead of overwriting
    messages: Annotated[list, add_messages]
    # Custom keys (per "Customize state" tutorial)
    name: str
    birthday: str

# ========== LLM (local, via Ollama) ==========
llm = ChatOllama(model=OLLAMA_MODEL)

# ========== Tools ==========
# -- SearXNG Web Search --
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

# -- Human Assistance (HITL) with state customization --
@tool("human_assistance")
def human_assistance(
    name: str,
    birthday: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Pause the graph and request human review of (name, birthday).
    Returns a Command(update=...) that updates the state and adds a ToolMessage.
    """
    # Ask for human confirmation/correction
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
            "hint": "Reply 'yes' to accept, or provide a JSON with corrected fields "
                    + '(e.g., {"name":"LangGraph","birthday":"Jan 17, 2024"}).',
        }
    )

    # Normalize human response:
    # - If it's a simple string like "yes"/"y", accept inputs.
    # - If it's JSON-like, pull corrected fields.
    verified_name = name
    verified_birthday = birthday
    response_text = ""

    try:
        # If response looks like JSON, parse it
        if isinstance(human_response, dict):
            # If a "data" key was used (resume={"data": "..."}), try to parse/inspect it
            payload = human_response.get("data", human_response)
            if isinstance(payload, str):
                # could be "yes" or a JSON string
                s = payload.strip()
                if s.lower().startswith(("y", "yes")):
                    response_text = "Correct"
                else:
                    try:
                        obj = json.loads(s)
                        verified_name = obj.get("name", name)
                        verified_birthday = obj.get("birthday", birthday)
                        response_text = f"Made a correction: {obj}"
                    except json.JSONDecodeError:
                        # free text correction not in JSON; just acknowledge
                        response_text = f"Human note: {s}"
            elif isinstance(payload, dict):
                # Dict with corrections
                if str(payload.get("correct", "")).lower().startswith(("y", "yes")):
                    response_text = "Correct"
                else:
                    verified_name = payload.get("name", name)
                    verified_birthday = payload.get("birthday", birthday)
                    response_text = f"Made a correction: {payload}"
        else:
            # treat anything else as simple text
            s = str(human_response).strip()
            if s.lower().startswith(("y", "yes")):
                response_text = "Correct"
            else:
                response_text = f"Human note: {s}"
    except Exception as e:
        # Fail open (don't block the flow)
        response_text = f"Human parsing error, proceeding with provided values: {e}"

    # Build a state update that:
    #  - sets name/birthday in the state
    #  - appends a ToolMessage back to the messages channel
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response_text, tool_call_id=tool_call_id)],
    }
    # Returning Command(update=...) tells LangGraph to apply this state update directly.
    return Command(update=state_update)

TOOLS = [web_search, human_assistance]

# LLM with tool “schema” bound so it knows how to call them.
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Graph nodes ==========
def chatbot(state: State) -> Dict[str, List]:
    """
    Calls the LLM with the running message list; may emit tool_calls.
    We keep tool calls to one at a time in HITL scenarios (interrupts pause mid-execution).
    """
    reply = llm_with_tools.invoke(state["messages"])
    if getattr(reply, "tool_calls", None):
        assert len(reply.tool_calls) <= 1, "Parallel tool calls disabled when using interrupts."
    return {"messages": [reply]}

class BasicToolNode:
    """
    Executes tool calls emitted by the last AI message.
    - If the tool returns a Command(update=...), return that Command so LangGraph updates
      the state (used by 'human_assistance').
    - Otherwise, wrap the tool's result into a ToolMessage as before.
    """
    def __init__(self, tools: List):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state")

        last = messages[-1]
        out_messages: List[ToolMessage] = []

        def _get(obj, key, default=""):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        tool_calls = getattr(last, "tool_calls", []) or []
        for tc in tool_calls:
            tool_name = _get(tc, "name")
            args = _get(tc, "args", {})
            tc_id = _get(tc, "id", "")  # many providers put the id here

            tool_obj = self.tools_by_name.get(tool_name)
            if tool_obj is None:
                out_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        name=tool_name or "unknown",
                        tool_call_id=tc_id,
                    )
                )
                continue

            try:
                # IMPORTANT: pass a full ToolCall payload (include BOTH 'id' and 'tool_call_id')
                invocation = {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": args,
                    "tool_call_id": tc_id,
                    "id": tc_id,
                }
                result = tool_obj.invoke(invocation)

                # If tool returns a Command (e.g., from human_assistance), let LangGraph apply it.
                if isinstance(result, Command):
                    return result

                payload = json.dumps(result, ensure_ascii=False)

            except GraphInterrupt:
                # Let LangGraph pause and persist; do NOT swallow this
                raise
            except Exception as e:
                payload = json.dumps({"error": f"Tool failed: {e}"}, ensure_ascii=False)

            out_messages.append(
                ToolMessage(
                    content=payload,
                    name=tool_name,
                    tool_call_id=tc_id,
                )
            )

        return {"messages": out_messages} if out_messages else {}
    
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

# Memory-enabled compilation (needed for interrupts + persistence)
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ========== Runner ==========
def stream_once_and_maybe_interrupt(user_input: Optional[str], thread_id: str) -> bool:
    """
    Send one user message (if provided) and stream assistant updates.
    Detect if execution is currently paused (interrupt) and return True if waiting for human input.
    """
    config = {"configurable": {"thread_id": thread_id}}
    if user_input is not None:
        initial_state = {"messages": [{"role": "user", "content": user_input}]}
        events = graph.stream(initial_state, config, stream_mode="values")
    else:
        events = graph.stream({}, config, stream_mode="values")

    waiting_for_human = False
    for event in events:
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

    # If paused at interrupt, snapshot.next will be set
    snapshot = graph.get_state(config)
    if snapshot.next:
        waiting_for_human = True
    return waiting_for_human

def resume_with_human_input(human_text: str, thread_id: str) -> None:
    """
    Resume a paused (interrupted) execution by sending a Command with data.
    Supports plain 'yes' or a JSON object with corrections.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Try to pass through structured corrections if human_text looks like JSON
    payload: Any
    s = human_text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            payload = {"data": s}
    else:
        payload = {"data": s}

    cmd = Command(resume=payload)
    events = graph.stream(cmd, config, stream_mode="values")
    for event in events:
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

def repl():
    print(f"[info] Using SearXNG at: {SEARXNG_HOST}")
    print(f"[info] Using Ollama model: {OLLAMA_MODEL}")
    print(f"[info] Using THREAD_ID: {THREAD_ID}  (set env THREAD_ID to change session)")
    print("[help] Type 'new:<id>' to switch threads, 'resume:<text>' to answer a pending human prompt.")
    print("[help] To correct state fields via resume, you can send JSON like: resume: {\"name\":\"LangGraph\",\"birthday\":\"Jan 17, 2024\"}")
    thread_id = THREAD_ID
    pending_interrupt = False

    while True:
        q = input("User: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q.lower().startswith("new:"):
            thread_id = q.split(":", 1)[1].strip() or thread_id
            print(f"[info] Switched to THREAD_ID={thread_id}")
            pending_interrupt = False
            continue
        if q.lower().startswith("resume:"):
            human_text = q.split(":", 1)[1].strip()
            if not human_text:
                print("[warn] Provide text after 'resume:'. Examples:")
                print("       resume: yes")
                print("       resume: {\"name\":\"LangGraph\",\"birthday\":\"Jan 17, 2024\"}")
                continue
            resume_with_human_input(human_text, thread_id)
            pending_interrupt = False
            continue

        pending_interrupt = stream_once_and_maybe_interrupt(q, thread_id)
        if pending_interrupt:
            print("[awaiting] Execution paused for human input (human_assistance).")
            print("[hint] Type: resume: yes")
            print('[hint] Or:   resume: {"name":"LangGraph","birthday":"Jan 17, 2024"}')

if __name__ == "__main__":
    repl()
