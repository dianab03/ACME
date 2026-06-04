from unittest.mock import MagicMock, patch

from app.assistant.client import MAX_ITERATIONS, OllamaClient


def _make_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", model="llama3.2")


def _make_resp(content=None, tool_calls=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    msg = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    resp.json.return_value = {"message": msg}
    return resp


def test_single_turn_no_tools_forces_tool_then_grounded_answer():
    no_tool_resp = _make_resp(content="I can answer directly.")
    tool_resp = _make_resp(tool_calls=[{"function": {"name": "list_instruments", "arguments": {}}}])
    final_resp = _make_resp(content="Done.")

    with patch("httpx.post", side_effect=[no_tool_resp, tool_resp, final_resp]):
        with patch(
            "app.assistant.client.execute_tool",
            return_value='[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',
        ):
            answer, tools_used = _make_client().chat("Question?", session=None)

    assert tools_used == ["list_instruments"]
    assert "Found 1 instruments" in answer
    assert "Grounding evidence:" in answer


def test_two_tool_calls_generate_analytics_summary():
    tool_resp_1 = _make_resp(tool_calls=[{"function": {"name": "list_instruments", "arguments": {}}}])
    tool_resp_2 = _make_resp(tool_calls=[{"function": {"name": "get_analytics", "arguments": {}}}])
    final_resp = _make_resp(content="All good.")

    with patch("httpx.post", side_effect=[tool_resp_1, tool_resp_2, final_resp]):
        with patch(
            "app.assistant.client.execute_tool",
            side_effect=[
                '[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',
                '{"count":10,"min_close":"1","max_close":"2","avg_close":"1.5","total_volume":100}',
            ],
        ):
            answer, tools_used = _make_client().chat("Give analytics summary.", session=MagicMock())

    assert tools_used == ["list_instruments", "get_analytics"]
    assert "Analytics summary:" in answer
    assert "avg_close=1.5" in answer
    assert "Grounding evidence:" in answer


def test_max_iterations_still_returns_grounded_payload():
    tool_resp = _make_resp(tool_calls=[{"function": {"name": "list_instruments", "arguments": {}}}])

    with patch("httpx.post", return_value=tool_resp):
        with patch(
            "app.assistant.client.execute_tool",
            return_value='[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',
        ):
            answer, tools_used = _make_client().chat("Loop forever", session=MagicMock())

    assert len(tools_used) == MAX_ITERATIONS
    assert "Found 1 instruments" in answer
