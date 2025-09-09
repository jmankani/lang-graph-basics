# LangGraph + Ollama + SearXNG Example

This repository contains a single Python script (`basic_chatbot.py`) that demonstrates how to build an **agentic chatbot** using [LangGraph](https://github.com/langchain-ai/langgraph), [Ollama](https://ollama.ai) for local LLM inference, and [SearXNG](https://docs.searxng.org/) for self-hosted web search.

The project progressively integrates all core LangGraph concepts:

- **Nodes & Edges** – simple stateful graph execution  
- **Tool calling** – use custom tools (e.g. web search, human assistance)  
- **Persistent memory** – resume sessions across turns  
- **Human-in-the-loop (HITL)** – pause for approval or correction mid-graph  
- **Customized state** – track arbitrary fields (`name`, `birthday`) alongside messages  
- **Time travel** – inspect history, replay past checkpoints, and fork alternative futures  

---

## Features

- **Local LLM** – runs entirely on your machine via Ollama (e.g. `mistral-nemo:12b`)  
- **Private Web Search** – queries routed through a Raspberry Pi–hosted SearXNG instance  
- **Tooling**:
  - `web_search` → fetch top results (title, url, snippet)  
  - `human_assistance` → pause execution until a human confirms/corrects state  
- **Interactive REPL** – type user messages, issue commands, and stream assistant responses  
- **Time Travel**:
  - `history` → list all checkpoints  
  - `replay:<index>` → resume from a past checkpoint  
  - `fork:<index> {overrides}` → branch into an alternate timeline with updated state  

---

## Requirements

- **Python** 3.9+ (tested with 3.11)  
- **Pipenv / venv** or any Python virtualenv  
- **Ollama** installed locally and running (see [docs](https://ollama.ai))  
- **SearXNG** instance reachable on your LAN (e.g. Raspberry Pi @ `http://192.168.x.x:8080`)  

Python deps are declared inline in the script but you’ll need:

```bash
pip install langchain-core langchain-ollama langgraph httpx typing_extensions
```

---

## Configuration

The script is configurable via environment variables:

| Variable         | Default                | Description                                |
|------------------|------------------------|--------------------------------------------|
| `SEARXNG_HOST`   | `http://192.168.68.63` | URL of your SearXNG instance               |
| `SEARXNG_TIMEOUT_S` | `30`                 | HTTP timeout for SearXNG queries           |
| `SEARXNG_RESULTS`  | `5`                  | Default number of search results per query |
| `OLLAMA_MODEL`   | `mistral-nemo:12b`     | Ollama model to use locally                |
| `THREAD_ID`      | `1`                    | Session / conversation ID                  |

Example (Linux/macOS):

```bash
export SEARXNG_HOST=http://192.168.1.42:8080
export OLLAMA_MODEL=mistral-nemo:12b
export THREAD_ID=42
```

---

## Usage

Run the REPL:

```bash
python basic_chatbot.py
```

You’ll see:

```
[info] Using SearXNG at: http://192.168.68.63
[info] Using Ollama model: mistral-nemo:12b
[info] Using THREAD_ID: 1  (set env THREAD_ID to change session)
[help] new:<id> | resume:<text> | history | replay:<index> | fork:<index> <json-overrides?>
```

### Normal chat
```
User: What is LangGraph?
Assistant: ...
```

### Web search
```
User: Use the web_search tool to find 3 results about "LangGraph tutorials"
```

### Human-in-the-loop
```
User: Please verify via human_assistance with name="LangGraph" and birthday="Jan 17, 2024".
```
→ Execution pauses until you type:
```
resume: yes
```
or
```
resume: {"name":"LG","birthday":"Jan 18, 2024"}
```

### Time travel
- `history` → list checkpoints  
- `replay:2` → resume execution from checkpoint index 2  
- `fork:3 {"name":"LG","birthday":"Jan 18, 2024"}` → branch timeline with overrides  

---

## Notes & Tips

- Ollama must be running (`ollama serve`) before starting the script.  
- For lighter hardware, you can swap the Ollama model (`OLLAMA_MODEL`) for a smaller one (e.g. `mistral:7b`).  
- SearXNG is optional; if unavailable, you can stub `web_search`.  
- Time travel features (`history`, `replay`, `fork`) rely on `InMemorySaver`. For persistence across process restarts, replace it with a DB-backed checkpointer.  
- Because we add system “policies” to discourage unnecessary tool calls, the assistant should answer simple profile questions (`What name and birthday do you currently have stored?`) directly from state instead of calling `human_assistance`.  

