"""
LangGraph + Ollama + SearXNG (hosted on Raspberry Pi)

Features
- Local LLM via Ollama (e.g., `mistral-nemo:12b`)
- Web search tool via SearXNG JSON API
- Tool-calling loop with LangGraph
- Persistent chat memory via InMemorySaver
- Human-in-the-loop via `interrupt` + `Command(update=..., return_value=...)`
- Customized state (name, birthday)
- Time travel: history, replay, fork with state edits

Run:
  python basic_chatbot.py

REPL tips:
  - Normal chat: just type your message
  - Switch session:   new:<id>
  - Pause/Resume HITL: resume:<text or JSON>
  - List history:      history
  - Replay checkpoint: replay:<index>
  - Fork checkpoint:   fork:<index> <json-overrides?>
    e.g., fork:2 {"name":"LG","birthday":"Jan 17, 2024"}
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

# HITL + time-travel primitives
from langgraph.types import Command, interrupt
from langgraph.errors import GraphInterrupt

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
    # Customized keys (lesson 5)
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
    Pause for human review; on resume, update state and emit a ToolMessage
    via the state update (no return_value).
    """
    human_response = interrupt({
        "question": "Is this correct?",
        "name": name,
        "birthday": birthday,
        "hint": (
            "Reply 'yes' to accept, or provide a JSON with corrected fields "
            '(e.g., {"name":"LangGraph","birthday":"Jan 17, 2024"}).'
        ),
    })

    # ---- normalize human response ----
    verified_name = name
    verified_birthday = birthday
    note = "Acknowledged"

    try:
        payload = human_response.get("data", human_response) if isinstance(human_response, dict) else human_response
        if isinstance(payload, str):
            s = payload.strip()
            if s.lower().startswith(("y", "yes")):
                note = "Correct"
            else:
                try:
                    obj = json.loads(s)
                    verified_name = obj.get("name", name)
                    verified_birthday = obj.get("birthday", birthday)
                    note = f"Made a correction: {obj}"
                except json.JSONDecodeError:
                    note = f"Human note: {s}"
        elif isinstance(payload, dict):
            if str(payload.get("correct", "")).lower().startswith(("y", "yes")):
                note = "Correct"
            else:
                verified_name = payload.get("name", name)
                verified_birthday = payload.get("birthday", birthday)
                note = f"Made a correction: {payload}"
        else:
            note = "Acknowledged"
    except Exception as e:
        note = f"Human parsing error, proceeding with provided values: {e}"

    # Build the state update:
    #  - update custom fields
    #  - append a ToolMessage so the LLM "sees" the human outcome
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [
            ToolMessage(
                content=json.dumps({
                    "status": "ok",
                    "note": note,
                    "name": verified_name,
                    "birthday": verified_birthday,
                }, ensure_ascii=False),
                name="human_assistance",
                tool_call_id=tool_call_id,
            )
        ],
    }
    return Command(update=state_update)

TOOLS = [web_search, human_assistance]

# LLM with tool schema bound so it knows how to call them.
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Graph nodes ==========
def chatbot(state: State) -> Dict[str, List]:
    """
    Calls the LLM with the running message list; may emit tool_calls.
    We keep tool calls to one at a time in HITL scenarios.
    """
    reply = llm_with_tools.invoke(state["messages"])
    if getattr(reply, "tool_calls", None):
        assert len(reply.tool_calls) <= 1, "Parallel tool calls disabled when using interrupts."
    return {"messages": [reply]}

class BasicToolNode:
    """
    Executes tool calls emitted by the last AI message.

    - Accepts full variety of tool returns:
      * Command(...)            -> returned directly (LangGraph applies it)
      * ToolMessage             -> appended directly
      * BaseMessage (any)       -> converted to ToolMessage(content=...)
      * Plain dict/list/str/... -> JSON-serialized (fallback default=str)
    - Correctly handles interrupts (GraphInterrupt).
    - Passes a FULL ToolCall dict (id + tool_call_id) for InjectedToolCallId.
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
            tc_id = _get(tc, "id", "")

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
                # IMPORTANT: pass full ToolCall payload
                invocation = {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": args,
                    "tool_call_id": tc_id,
                    "id": tc_id,
                }
                result = tool_obj.invoke(invocation)

                # 1) Command -> hand back to LangGraph
                if isinstance(result, Command):
                    return result

                # 2) ToolMessage -> append as-is
                if isinstance(result, ToolMessage):
                    out_messages.append(result)
                    continue

                # 3) Any BaseMessage -> convert to ToolMessage
                if isinstance(result, BaseMessage):
                    out_messages.append(
                        ToolMessage(
                            content=result.content,
                            name=tool_name,
                            tool_call_id=tc_id,
                        )
                    )
                    continue

                # 4) Plain python types -> JSON (safe fallback)
                try:
                    payload = json.dumps(result, ensure_ascii=False)
                except Exception:
                    payload = json.dumps(result, ensure_ascii=False, default=str)

            except GraphInterrupt:
                # Pause here; LangGraph will persist and resume later
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

# Memory-enabled compilation (needed for interrupts + persistence + time travel)
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ========== Runner helpers ==========
def _config_for(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}

def print_history(thread_id: str) -> list:
    """List checkpoints (reverse-chronological). Returns the list for further use."""
    config = _config_for(thread_id)
    states = list(graph.get_state_history(config))  # newest first
    if not states:
        print("[history] No checkpoints yet.")
        return []
    print("\n[history] index | next_nodes     | checkpoint_id")
    for idx, s in enumerate(states):
        nxt = ",".join(s.next) if s.next else "-"
        cpid = s.config["configurable"].get("checkpoint_id", "")
        print(f"[history] {idx:>5} | {nxt:<14} | {cpid}")
    print()
    return states

def replay_from_index(thread_id: str, index: int) -> None:
    """Resume execution from a past checkpoint (no state edits)."""
    states = print_history(thread_id)
    if not states:
        return
    if index < 0 or index >= len(states):
        print(f("[replay] Invalid index {index}."))
        return
    selected = states[index]
    config = selected.config  # contains thread_id + checkpoint_id
    print(f"[replay] Resuming from checkpoint_id={config['configurable']['checkpoint_id']}")
    for event in graph.stream(None, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

def fork_from_index(thread_id: str, index: int, overrides: Optional[Dict[str, Any]]) -> None:
    """
    Create a new checkpoint by editing state, then resume from it.
    Overrides can edit 'name', 'birthday', or even append an injected message:
      {"name": "LG", "birthday": "Jan 17, 2024"}
      {"messages": [{"role":"system","content":"Use web_search for links."}]}
    """
    states = print_history(thread_id)
    if not states:
        return
    if index < 0 or index >= len(states):
        print(f("[fork] Invalid index {index}."))
        return
    selected = states[index]
    base_config = selected.config
    values = overrides or {}
    new_config = graph.update_state(base_config, values=values)
    print(f"[fork] Created new checkpoint_id={new_config['configurable']['checkpoint_id']} with overrides={values or {}}")
    for event in graph.stream(None, new_config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

# ========== Runner ==========
def stream_once_and_maybe_interrupt(user_input: Optional[str], thread_id: str) -> bool:
    """
    Send one user message (if provided) and stream assistant updates.
    Detect if execution is currently paused (interrupt) and return True if waiting for human input.
    """
    config = _config_for(thread_id)
    if user_input is not None:
        initial_state = {"messages": [{"role": "user", "content": user_input}]}
        events = graph.stream(initial_state, config, stream_mode="values")
    else:
        events = graph.stream({}, config, stream_mode="values")

    waiting_for_human = False
    for event in events:
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

    snapshot = graph.get_state(config)
    if snapshot.next:
        waiting_for_human = True
    return waiting_for_human

def resume_with_human_input(human_text: str, thread_id: str) -> None:
    """
    Resume a paused (interrupted) execution by sending a Command with data.
    Supports plain 'yes' or a JSON object with corrections.
    """
    config = _config_for(thread_id)
    s = human_text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            payload: Any = json.loads(s)
        except json.JSONDecodeError:
            payload = {"data": s}
    else:
        payload = {"data": s}

    cmd = Command(resume=payload)
    for event in graph.stream(cmd, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            print("Assistant:", event["messages"][-1].content)

def repl():
    print(f"[info] Using SearXNG at: {SEARXNG_HOST}")
    print(f"[info] Using Ollama model: {OLLAMA_MODEL}")
    print(f"[info] Using THREAD_ID: {THREAD_ID}  (set env THREAD_ID to change session)")
    print("[help] new:<id> | resume:<text> | history | replay:<index> | fork:<index> <json-overrides?>")
    print('[help] Example fork: fork:2 {"name":"LG","birthday":"Jan 17, 2024"}')
    thread_id = THREAD_ID

    while True:
        q = input("User: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break

        # Commands
        if q.lower().startswith("new:"):
            thread_id = q.split(":", 1)[1].strip() or thread_id
            print(f"[info] Switched to THREAD_ID={thread_id}")
            continue

        if q.lower() == "history":
            print_history(thread_id)
            continue

        if q.lower().startswith("replay:"):
            try:
                idx = int(q.split(":", 1)[1].strip())
            except ValueError:
                print("[replay] Usage: replay:<index>")
                continue
            replay_from_index(thread_id, idx)
            continue

        if q.lower().startswith("fork:"):
            rest = q.split(":", 1)[1].strip()
            if " " in rest:
                idx_str, json_str = rest.split(" ", 1)
            else:
                idx_str, json_str = rest, ""
            try:
                idx = int(idx_str)
            except ValueError:
                print("[fork] Usage: fork:<index> {optional JSON overrides}")
                continue
            overrides = None
            if json_str:
                try:
                    overrides = json.loads(json_str)
                except json.JSONDecodeError:
                    print("[fork] Invalid JSON overrides. Example: {'name':'LG','birthday':'Jan 17, 2024'}")
                    continue
            fork_from_index(thread_id, idx, overrides)
            continue

        if q.lower().startswith("resume:"):
            human_text = q.split(":", 1)[1].strip()
            if not human_text:
                print("[resume] Provide text after 'resume:'. Examples:")
                print("         resume: yes")
                print("         resume: {\"name\":\"LangGraph\",\"birthday\":\"Jan 17, 2024\"}")
                continue
            resume_with_human_input(human_text, thread_id)
            continue

        # Normal turn
        pending_interrupt = stream_once_and_maybe_interrupt(q, thread_id)
        if pending_interrupt:
            print("[awaiting] Execution paused for human input (human_assistance).")
            print("[hint] Type: resume: yes")
            print('[hint] Or:   resume: {"name":"LangGraph","birthday":"Jan 17, 2024"}')

if __name__ == "__main__":
    repl()
