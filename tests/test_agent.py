import json
from types import SimpleNamespace

from research_assistant.agent import format_trace, run_agent, summarize_result


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def reply(content="", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Returns scripted replies and records every request."""

    def __init__(self, replies, final="final answer"):
        self.replies = list(replies)
        self.final = final
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("tool_choice") == "none":
            return reply(self.final)
        return self.replies.pop(0)


class FakeToolbox:
    schemas = []

    def __init__(self):
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return json.dumps({"results": [{"source": "arXiv:1, p.2", "title": "Paper"}]})


def test_agent_runs_tool_then_answers():
    client = FakeClient([
        reply(tool_calls=[tool_call("c1", "search_index", '{"query": "self-rag"}')]),
        reply("It retrieves on demand [arXiv:1, p.2]"),
    ])
    toolbox = FakeToolbox()
    result = run_agent("question?", toolbox, client, "model")

    assert result.answer.startswith("It retrieves")
    assert toolbox.calls == [("search_index", {"query": "self-rag"})]
    assert len(result.steps) == 1 and not result.stopped_early
    last_message = client.requests[1]["messages"][-1]
    assert last_message["role"] == "tool" and last_message["tool_call_id"] == "c1"


def test_agent_stops_at_step_limit():
    looping = [reply(tool_calls=[tool_call(f"c{i}", "search_index", "{}")]) for i in range(5)]
    client = FakeClient(looping, final="best effort answer")
    result = run_agent("question?", FakeToolbox(), client, "model", max_steps=2)

    assert result.stopped_early
    assert len(result.steps) == 2
    assert result.answer == "best effort answer"
    assert client.requests[-1]["tool_choice"] == "none"


def test_invalid_arguments_are_reported_not_executed():
    client = FakeClient([
        reply(tool_calls=[tool_call("c1", "search_index", "not json")]),
        reply("done"),
    ])
    toolbox = FakeToolbox()
    result = run_agent("question?", toolbox, client, "model")

    assert toolbox.calls == []
    assert result.steps[0].summary.startswith("error")


def test_rate_limit_is_retried(monkeypatch):
    class FakeRateLimit(Exception):
        pass

    monkeypatch.setattr("research_assistant.agent.time.sleep", lambda seconds: None)
    client = FakeClient([reply("answer")])
    original = client.create
    attempts = []

    def flaky_create(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise FakeRateLimit()
        return original(**kwargs)

    client.chat.completions.create = flaky_create
    result = run_agent("q", FakeToolbox(), client, "model", retry_on=(FakeRateLimit,))
    assert result.answer == "answer" and len(attempts) == 2


def test_summarize_result():
    assert summarize_result('{"error": "boom"}') == "error: boom"
    assert summarize_result('{"added": [{"arxiv_id": "1", "chunks": 9}], "errors": []}') == (
        "indexed 1 (9 chunks)"
    )
    assert summarize_result('{"results": [], "note": "empty"}') == "empty"
    assert summarize_result("not json") == "invalid tool output"


def test_format_trace_contains_steps_and_answer():
    client = FakeClient([
        reply(tool_calls=[tool_call("c1", "list_indexed_papers", "{}")]),
        reply("final | answer"),
    ])
    result = run_agent("q", FakeToolbox(), client, "model")
    trace = format_trace("q", result, "model")
    assert "`list_indexed_papers`" in trace and "final | answer" in trace
