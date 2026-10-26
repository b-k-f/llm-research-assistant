"""Stage 3: a research agent that decides which tools to call.

A plain Python tool-calling loop (no framework):
  1. send the conversation and the tool schemas to the LLM
  2. if the LLM asks for tool calls, run them and append the results to the conversation
  3. repeat until the LLM answers without calling a tool, or the step budget runs out
"""
import json
import time
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are a research agent that answers questions about scientific papers.
You have tools to search arXiv, add papers to a full-text index, and search that index.

How to work:
1. Check what is already indexed (list_indexed_papers or search_index).
2. If the index lacks the information, use search_arxiv. Rewrite the question into a good
   arXiv query and refine it if the results look off-topic.
3. Add only the most relevant papers (at most 3) with add_papers.
4. Use search_index with focused queries to find the passages you need.
5. Answer using only passages returned by search_index. Cite every claim with its source,
   e.g. [arXiv:2310.11511v1, p.3]. Never answer from memory.
If you cannot find the answer in the papers, say so. Be efficient with tool calls."""

FINAL_ANSWER_PROMPT = (
    "You have used all your steps. Write your final answer now, using only the information "
    "gathered so far, with citations."
)


@dataclass
class Step:
    number: int
    tool: str
    arguments: dict
    summary: str


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_early: bool = False


def summarize_result(result_json: str) -> str:
    """One-line summary of a tool result, for the trace."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return "invalid tool output"
    if "error" in data:
        return f"error: {data['error']}"
    if "added" in data:
        parts = [f"{p['arxiv_id']} ({p['chunks']} chunks)" for p in data["added"]]
        summary = "indexed " + (", ".join(parts) or "nothing")
        if data.get("errors"):
            summary += f"; {len(data['errors'])} error(s)"
        return summary
    if "papers" in data:
        return f"{len(data['papers'])} papers, {data.get('chunks', 0)} chunks in index"
    if "results" in data:
        results = data["results"]
        if not results:
            return data.get("note", "no results")
        top = results[0].get("title") or results[0].get("source", "")
        return f"{len(results)} results, top: {top}"[:150]
    return "ok"


def _complete(client, retry_on: tuple, retries: int = 4, **kwargs):
    """Call the LLM, waiting and retrying on rate-limit errors."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except retry_on:
            wait = 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM API still rate limited after several retries")


def run_agent(question: str, toolbox, client, model: str, max_steps: int = 8,
              retry_on: tuple = ()) -> AgentResult:
    """Run the tool-calling loop. `max_steps` limits the number of LLM turns that call tools."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []

    for _ in range(max_steps):
        response = _complete(client, retry_on, model=model, messages=messages,
                             tools=toolbox.schemas, temperature=0)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return AgentResult(answer=message.content or "", steps=steps)

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
                for call in tool_calls
            ],
        })
        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if isinstance(args, dict):
                result = toolbox.call(call.function.name, args)
            else:
                args = {}
                result = json.dumps({"error": "arguments must be a JSON object"})

            step = Step(len(steps) + 1, call.function.name, args, summarize_result(result))
            steps.append(step)
            print(f"  step {step.number}: {step.tool}({json.dumps(args)}) -> {step.summary}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Step budget used up: ask for a final answer without allowing more tool calls.
    messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
    response = _complete(client, retry_on, model=model, messages=messages,
                         tools=toolbox.schemas, tool_choice="none", temperature=0)
    return AgentResult(answer=response.choices[0].message.content or "", steps=steps,
                       stopped_early=True)


def _cell(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_trace(question: str, result: AgentResult, model: str) -> str:
    """Markdown report of the agent run: every tool call, then the final answer."""
    lines = [
        "## Agent run",
        "",
        f"**Question:** {question}",
        f"**Model:** `{model}` - **Tool calls:** {len(result.steps)}",
        "",
        "| Step | Tool | Arguments | Result |",
        "|---|---|---|---|",
    ]
    for step in result.steps:
        lines.append(
            f"| {step.number} | `{step.tool}` | {_cell(json.dumps(step.arguments))} | "
            f"{_cell(step.summary)} |"
        )
    lines += ["", "### Answer", "", result.answer.strip() or "_(empty answer)_"]
    if result.stopped_early:
        lines += ["", "_The agent reached its step limit before finishing._"]
    return "\n".join(lines) + "\n"
