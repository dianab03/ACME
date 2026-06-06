import json
from datetime import date
from uuid import UUID

import httpx

from app.assistant.tools import TOOL_SCHEMAS, execute_tool

MAX_ITERATIONS = 6


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url
        self._model = model

    def chat(self, question: str, session) -> tuple[str, list[str]]:
        """
        Run the agentic tool-calling loop against Ollama.
        Returns (answer, tool_calls_used).
        Raises httpx.ConnectError if Ollama is unreachable.
        """
        system_prompt = (
            "You are a data-warehouse assistant. "
            "Always use tools and answer ONLY from tool outputs. "
            "Never invent UUIDs, prices, counts, dates, or providers. "
            "If IDs are missing, call list_instruments and/or list_sources first. "
            "If a tool fails, recover with another relevant tool call."
        )
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        tool_calls_used: list[str] = []
        runs: list[dict] = []
        cached_instrument_ids: list[str] = []
        cached_source_ids: list[str] = []
        symbol_to_id_map: dict[str, str] = {}

        # Pre-populate with list_instruments and list_sources so LLM has proper IDs from the start
        self._preload_base_lists(session, runs, tool_calls_used, cached_instrument_ids, cached_source_ids)
        
        # Build symbol->ID map from preloaded instruments and add to context
        symbol_to_id_map = self._build_symbol_map(runs)
        context_msg = self._build_context_message(question, symbol_to_id_map, cached_instrument_ids, cached_source_ids)
        messages.append({"role": "user", "content": context_msg})

        for _ in range(MAX_ITERATIONS):
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            msg = body.get("message")
            if msg is None:
                raise ValueError(f"Unexpected Ollama response shape: {body}")

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                if len([n for n in [r["name"] for r in runs] if n not in ["list_instruments", "list_sources"]]) == 0:
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": "You must call at least one query tool (besides list_instruments and list_sources) before answering.",
                    })
                    continue
                break

            messages.append(msg)
            for tc in tool_calls:
                call_id = tc.get("id", "")
                name = tc["function"]["name"]
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                args = self._sanitize_args(name, args, cached_instrument_ids, cached_source_ids, question, symbol_to_id_map)

                raw = execute_tool(name, args, session)
                parsed, err = self._parse_tool_output(raw)
                self._update_caches(name, parsed, cached_instrument_ids, cached_source_ids)

                # Detect insufficient data and flag it as an error
                if err is None:
                    err = self._check_insufficient_data(name, parsed)

                tool_calls_used.append(name)
                runs.append({
                    "name": name,
                    "args": args,
                    "raw": raw,
                    "data": parsed,
                    "error": err,
                })
                messages.append({"role": "tool", "tool_call_id": call_id, "content": raw})

        self._run_fallback_tools(question, session, runs, tool_calls_used, cached_instrument_ids, cached_source_ids, symbol_to_id_map)
        answer = self._build_grounded_answer(question, runs)
        return answer, tool_calls_used

    @staticmethod
    def _parse_tool_output(raw: str) -> tuple[dict | list | None, str | None]:
        try:
            payload = json.loads(raw)
        except Exception:
            return None, "Tool returned non-JSON content."
        if isinstance(payload, dict) and payload.get("error"):
            return payload, str(payload.get("error"))
        return payload, None

    @staticmethod
    def _check_insufficient_data(name: str, payload: dict | list | None) -> str | None:
        """Detect when a tool returned success but with insufficient/empty data."""
        if name in ["get_trend", "get_analytics", "get_forecast", "get_risk_signal"]:
            if isinstance(payload, dict):
                # Check if all value fields are None or missing
                if name == "get_trend":
                    if (payload.get("start_close") is None and payload.get("end_close") is None 
                        and payload.get("change") is None and payload.get("change_pct") is None):
                        return "Insufficient data (no valid price records in date range)"
                elif name == "get_analytics":
                    if payload.get("count") == 0 or payload.get("avg_close") is None:
                        return "Insufficient data (no records in date range)"
                elif name == "get_forecast":
                    if payload.get("predicted_next_close") is None:
                        return "Insufficient data for forecast"
                elif name == "get_risk_signal":
                    if payload.get("signal") == "unknown":
                        return "Insufficient data for risk calculation"
        elif name == "get_time_series":
            if isinstance(payload, list) and len(payload) == 0:
                return "No time series data found"
        return None

    @staticmethod
    def _is_valid_uuid(value: str | None) -> bool:
        if not value:
            return False
        try:
            UUID(str(value))
            return True
        except Exception:
            return False

    def _sanitize_args(
        self,
        name: str,
        args: dict,
        cached_instrument_ids: list[str],
        cached_source_ids: list[str],
        question: str = "",
        symbol_map: dict[str, str] | None = None,
    ) -> dict:
        symbol_map = symbol_map or {}
        out = dict(args or {})
        start = out.get("start_date") or "2000-01-01"
        end = out.get("end_date") or date.today().isoformat()

        if name == "get_instrument":
            if not self._is_valid_uuid(out.get("instrument_id")) and cached_instrument_ids:
                # Try to find instrument by symbol from question
                instrument_id = self._find_instrument_id_from_question(question, symbol_map)
                out["instrument_id"] = instrument_id or cached_instrument_ids[0]

        if name in {"get_time_series", "get_analytics", "get_trend", "get_forecast", "get_risk_signal"}:
            if not self._is_valid_uuid(out.get("instrument_id")) and cached_instrument_ids:
                # Try to find instrument by symbol from question
                instrument_id = self._find_instrument_id_from_question(question, symbol_map)
                out["instrument_id"] = instrument_id or cached_instrument_ids[0]
            if not self._is_valid_uuid(out.get("source_id")) and cached_source_ids:
                out["source_id"] = cached_source_ids[0]
            out["start_date"] = start
            out["end_date"] = end

        if name == "compare_assets":
            if not self._is_valid_uuid(out.get("source_id")) and cached_source_ids:
                out["source_id"] = cached_source_ids[0]
            if not self._is_valid_uuid(out.get("instrument_a_id")) and cached_instrument_ids:
                instrument_id = self._find_instrument_id_from_question(question, symbol_map)
                out["instrument_a_id"] = instrument_id or cached_instrument_ids[0]
            if not self._is_valid_uuid(out.get("instrument_b_id")) and len(cached_instrument_ids) > 1:
                out["instrument_b_id"] = cached_instrument_ids[1]
            out["start_date"] = start
            out["end_date"] = end

        return out

    def _find_instrument_id_from_question(self, question: str, symbol_map: dict[str, str]) -> str | None:
        """Extract an instrument symbol from the question and return its ID."""
        if not question or not symbol_map:
            return None
        
        # Convert question to uppercase for symbol matching
        question_upper = question.upper()
        
        # Check for exact symbol matches (e.g., "MSFT", "AAPL")
        for symbol in symbol_map.keys():
            if symbol in question_upper:
                return symbol_map[symbol]
        
        return None

    def _preload_base_lists(
        self,
        session,
        runs: list[dict],
        tool_calls_used: list[str],
        cached_instrument_ids: list[str],
        cached_source_ids: list[str],
    ) -> None:
        """Pre-load list_instruments and list_sources before the main loop."""
        for tool_name in ["list_instruments", "list_sources"]:
            raw = execute_tool(tool_name, {}, session)
            parsed, err = self._parse_tool_output(raw)
            self._update_caches(tool_name, parsed, cached_instrument_ids, cached_source_ids)
            tool_calls_used.append(tool_name)
            runs.append({"name": tool_name, "args": {}, "raw": raw, "data": parsed, "error": err})

    def _build_symbol_map(self, runs: list[dict]) -> dict[str, str]:
        """Extract symbol-to-ID mapping from preloaded list_instruments."""
        symbol_map = {}
        for run in runs:
            if run["name"] == "list_instruments" and isinstance(run["data"], list):
                for item in run["data"]:
                    if isinstance(item, dict):
                        symbol = item.get("symbol", "").upper()
                        instrument_id = str(item.get("instrument_id", ""))
                        if symbol and instrument_id:
                            symbol_map[symbol] = instrument_id
        return symbol_map

    def _build_context_message(self, question: str, symbol_map: dict[str, str], cached_instrument_ids: list[str], cached_source_ids: list[str]) -> str:
        """Build a context message that includes the question plus available symbols and sources."""
        context_parts = [question]
        
        if symbol_map:
            symbols_list = ", ".join(sorted(symbol_map.keys()))
            context_parts.append(f"\nAvailable instruments: {symbols_list}")
        
        if cached_source_ids:
            context_parts.append(f"\nData sources available: {len(cached_source_ids)}")
        
        return "\n".join(context_parts)

    @staticmethod
    def _update_caches(
        name: str,
        payload: dict | list | None,
        cached_instrument_ids: list[str],
        cached_source_ids: list[str],
    ) -> None:
        if name == "list_instruments" and isinstance(payload, list):
            ids = [str(item.get("instrument_id")) for item in payload if isinstance(item, dict) and item.get("instrument_id")]
            if ids:
                cached_instrument_ids.clear()
                cached_instrument_ids.extend(ids)
        elif name == "list_sources" and isinstance(payload, list):
            ids = [str(item.get("source_id")) for item in payload if isinstance(item, dict) and item.get("source_id")]
            if ids:
                cached_source_ids.clear()
                cached_source_ids.extend(ids)

    def _run_fallback_tools(
        self,
        question: str,
        session,
        runs: list[dict],
        tool_calls_used: list[str],
        cached_instrument_ids: list[str],
        cached_source_ids: list[str],
        symbol_to_id_map: dict[str, str] | None = None,
    ) -> None:
        symbol_to_id_map = symbol_to_id_map or {}
        q = question.lower()
        # Only count successful calls (no error)
        successful_names = [r["name"] for r in runs if r.get("error") is None]
        # Also look for calls that had insufficient data (marked as error by _check_insufficient_data)
        all_names = [r["name"] for r in runs]

        def call(name: str, args: dict):
            safe_args = self._sanitize_args(name, args, cached_instrument_ids, cached_source_ids, question, symbol_to_id_map)
            raw = execute_tool(name, safe_args, session)
            parsed, err = self._parse_tool_output(raw)
            self._update_caches(name, parsed, cached_instrument_ids, cached_source_ids)
            if err is None:
                err = self._check_insufficient_data(name, parsed)
            tool_calls_used.append(name)
            runs.append({"name": name, "args": safe_args, "raw": raw, "data": parsed, "error": err})

        # list_instruments and list_sources should already be pre-loaded, skip
        if "list_instruments" not in all_names:
            call("list_instruments", {})
            successful_names = [r["name"] for r in runs if r.get("error") is None]

        needs_source = any(k in q for k in ["analytics", "average", "trend", "forecast", "risk", "time series", "compare"])
        if needs_source and "list_sources" not in all_names:
            call("list_sources", {})
            successful_names = [r["name"] for r in runs if r.get("error") is None]

        # Retry tools that had insufficient data or weren't called
        if any(k in q for k in ["average close", "avg close", "mean close", "analytics"]) and "get_analytics" not in successful_names:
            call("get_analytics", {})

        if any(k in q for k in ["trend", "explain a change", "summarize"]) and "get_trend" not in successful_names:
            call("get_trend", {})

        if any(k in q for k in ["forecast", "next day", "predict"]) and "get_forecast" not in successful_names:
            call("get_forecast", {})

        if any(k in q for k in ["risk", "drawdown", "volatility"]) and "get_risk_signal" not in successful_names:
            call("get_risk_signal", {})

        if "compare" in q and "compare_assets" not in successful_names and len(cached_instrument_ids) >= 2:
            call("compare_assets", {})

    def _build_grounded_answer(self, question: str, runs: list[dict]) -> str:
        successful = [r for r in runs if r.get("error") is None]
        if not successful:
            return "I could not produce a grounded answer because all tool calls failed."

        lines: list[str] = []
        latest = {r["name"]: r for r in successful}

        li = latest.get("list_instruments")
        if li and isinstance(li["data"], list):
            items = li["data"]
            lines.append(f"Found {len(items)} instruments in the warehouse.")
            if items:
                preview = ", ".join(
                    f"{i.get('symbol', '?')} ({i.get('instrument_id', '?')})"
                    for i in items[:5]
                    if isinstance(i, dict)
                )
                if preview:
                    lines.append(f"Sample instruments: {preview}.")

        ga = latest.get("get_analytics")
        if ga and isinstance(ga["data"], dict):
            d = ga["data"]
            lines.append(
                "Analytics summary: "
                f"count={d.get('count')}, min_close={d.get('min_close')}, "
                f"max_close={d.get('max_close')}, avg_close={d.get('avg_close')}, "
                f"total_volume={d.get('total_volume')}."
            )

        gt = latest.get("get_trend")
        if gt and isinstance(gt["data"], dict):
            d = gt["data"]
            lines.append(
                "Trend summary: "
                f"direction={d.get('direction')}, change={d.get('change')}, change_pct={d.get('change_pct')}."
            )

        gf = latest.get("get_forecast")
        if gf and isinstance(gf["data"], dict):
            d = gf["data"]
            lines.append(
                f"Forecast summary: method={d.get('method')}, predicted_next_close={d.get('predicted_next_close')}."
            )

        gr = latest.get("get_risk_signal")
        if gr and isinstance(gr["data"], dict):
            d = gr["data"]
            lines.append(
                "Risk summary: "
                f"signal={d.get('signal')}, volatility_pct={d.get('volatility_pct')}, "
                f"max_drawdown_pct={d.get('max_drawdown_pct')}."
            )

        gc = latest.get("compare_assets")
        if gc and isinstance(gc["data"], dict):
            d = gc["data"]
            lines.append(
                "Comparison summary: "
                f"instrument_a_count={d.get('instrument_a_count')}, instrument_b_count={d.get('instrument_b_count')}, "
                f"instrument_a_avg_close={d.get('instrument_a_avg_close')}, "
                f"instrument_b_avg_close={d.get('instrument_b_avg_close')}, avg_close_diff={d.get('avg_close_diff')}."
            )

        gts = latest.get("get_time_series")
        if gts and isinstance(gts["data"], list):
            rows = gts["data"]
            lines.append(f"Fetched {len(rows)} time-series rows.")

        if not lines:
            lines.append("Tool calls completed, but no usable payload was returned.")

        evidence = ["Grounding evidence:"]
        for r in successful[-4:]:
            snippet = str(r["raw"]).replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            evidence.append(f"- {r['name']}: {snippet}")

        failed = [r for r in runs if r.get("error")]
        if failed:
            lines.append("Some tool calls failed and were ignored for the final summary.")

        return "\n".join(lines + [""] + evidence)
