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
    tool_resp = _make_resp(tool_calls=[{"function": {"name": "get_analytics", "arguments": {}}}])
    final_resp = _make_resp(content="Done.")

    # Preload calls list_instruments and list_sources first, then main loop runs
    with patch("httpx.post", side_effect=[no_tool_resp, tool_resp, final_resp]):
        with patch(
            "app.assistant.client.execute_tool",
            side_effect=[
                '[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',  # preload list_instruments
                '[{"source_id":"00000000-0000-0000-0000-000000000002","name":"NASDAQ","type":"rest"}]',  # preload list_sources
                '{"count":5,"min_close":"100","max_close":"150","avg_close":"125","total_volume":50000}',  # get_analytics
            ]
        ):
            answer, tools_used = _make_client().chat("Question?", session=None)

    assert "list_instruments" in tools_used
    assert "list_sources" in tools_used
    assert "get_analytics" in tools_used
    assert "Analytics summary:" in answer
    assert "Grounding evidence:" in answer


def test_two_tool_calls_generate_analytics_summary():
    tool_resp_1 = _make_resp(tool_calls=[{"function": {"name": "get_analytics", "arguments": {}}}])
    final_resp = _make_resp(content="All good.")

    # Preload calls list_instruments and list_sources first
    with patch("httpx.post", side_effect=[tool_resp_1, final_resp]):
        with patch(
            "app.assistant.client.execute_tool",
            side_effect=[
                '[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',  # preload list_instruments
                '[{"source_id":"00000000-0000-0000-0000-000000000002","name":"NASDAQ","type":"rest"}]',  # preload list_sources
                '{"count":10,"min_close":"1","max_close":"2","avg_close":"1.5","total_volume":100}',  # get_analytics
            ],
        ):
            answer, tools_used = _make_client().chat("Give analytics summary.", session=MagicMock())

    assert "list_instruments" in tools_used
    assert "list_sources" in tools_used
    assert "get_analytics" in tools_used
    assert "Analytics summary:" in answer
    assert "avg_close=1.5" in answer
    assert "Grounding evidence:" in answer


def test_max_iterations_still_returns_grounded_payload():
    tool_resp = _make_resp(tool_calls=[{"function": {"name": "get_analytics", "arguments": {}}}])

    # Preload runs twice, then MAX_ITERATIONS times we get tool_resp
    responses = [tool_resp] * MAX_ITERATIONS
    with patch("httpx.post", side_effect=responses):
        with patch(
            "app.assistant.client.execute_tool",
            side_effect=[
                '[{"instrument_id":"00000000-0000-0000-0000-000000000001","symbol":"MSFT"}]',  # preload list_instruments
                '[{"source_id":"00000000-0000-0000-0000-000000000002","name":"NASDAQ","type":"rest"}]',  # preload list_sources
            ] + ['{"count":0,"min_close":null,"max_close":null,"avg_close":null,"total_volume":0}'] * MAX_ITERATIONS,  # get_analytics repeated
        ):
            answer, tools_used = _make_client().chat("Loop forever", session=MagicMock())

    # Should have list_instruments, list_sources, and then MAX_ITERATIONS of get_analytics
    assert tools_used[0] == "list_instruments"
    assert tools_used[1] == "list_sources"
    assert "Found 1 instruments" in answer


def test_symbol_extraction_finds_correct_instrument():
    """Test that _find_instrument_id_from_question correctly extracts symbol from question."""
    client = _make_client()
    
    symbol_map = {
        "MSFT": "00000000-0000-0000-0000-000000000001",
        "AAPL": "00000000-0000-0000-0000-000000000002",
        "DEMO": "00000000-0000-0000-0000-000000000003",
    }
    
    # Test exact symbol match
    result = client._find_instrument_id_from_question("Summarize the trend for MSFT.", symbol_map)
    assert result == "00000000-0000-0000-0000-000000000001"
    
    # Test case-insensitive match
    result = client._find_instrument_id_from_question("what is the trend for msft?", symbol_map)
    assert result == "00000000-0000-0000-0000-000000000001"
    
    # Test multiple symbols (should find first match)
    result = client._find_instrument_id_from_question("compare MSFT and AAPL", symbol_map)
    assert result in ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"]
    
    # Test no match
    result = client._find_instrument_id_from_question("show me the data", symbol_map)
    assert result is None


def test_symbol_map_building():
    """Test that symbol_map is correctly built from preloaded instruments."""
    tool_resp = _make_resp(tool_calls=[{"function": {"name": "get_trend", "arguments": {}}}])
    final_resp = _make_resp(content="Done.")
    
    with patch("httpx.post", side_effect=[tool_resp, final_resp]):
        with patch(
            "app.assistant.client.execute_tool",
            side_effect=[
                '[{"instrument_id":"id-msft","symbol":"MSFT","name":"Microsoft"},{"instrument_id":"id-aapl","symbol":"AAPL","name":"Apple"}]',  # preload list_instruments
                '[]',  # preload list_sources
                '{"direction":"up","start_close":"100","end_close":"150","change":"50","change_pct":"50"}',  # get_trend
            ]
        ):
            client = _make_client()
            answer, tools_used = client.chat("What is the trend for MSFT?", session=MagicMock())
    
    assert "get_trend" in tools_used
    # The answer should contain trend data, not null values
    assert "direction=up" in answer or "Trend summary:" in answer
