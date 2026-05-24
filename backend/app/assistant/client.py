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
            {"role": "user", "content": question},
        ]
        tool_calls_used: list[str] = []
        runs: list[dict] = []
        cached_instrument_ids: list[str] = []
        cached_source_ids: list[str] = []

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
                if not tool_calls_used:
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": "You must call at least one tool before answering.",
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
                args = self._sanitize_args(name, args, cached_instrument_ids, cached_source_ids)

                raw = execute_tool(name, args, session)
                parsed, err = self._parse_tool_output(raw)
                self._update_caches(name, parsed, cached_instrument_ids, cached_source_ids)

                tool_calls_used.append(name)
                runs.append({
                    "name": name,
                    "args": args,
                    "raw": raw,
                    "data": parsed,
                    "error": err,
                })
                messages.append({"role": "tool", "tool_call_id": call_id, "content": raw})

        self._run_fallback_tools(question, session, runs, tool_calls_used, cached_instrument_ids, cached_source_ids)
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
    ) -> dict:
        out = dict(args or {})
        start = out.get("start_date") or "2000-01-01"
        end = out.get("end_date") or date.today().isoformat()

        if name == "get_instrument":
            if not self._is_valid_uuid(out.get("instrument_id")) and cached_instrument_ids:
                out["instrument_id"] = cached_instrument_ids[0]

        if name in {"get_time_series", "get_analytics", "get_trend", "get_forecast", "get_risk_signal"}:
            if not self._is_valid_uuid(out.get("instrument_id")) and cached_instrument_ids:
                out["instrument_id"] = cached_instrument_ids[0]
            if not self._is_valid_uuid(out.get("source_id")) and cached_source_ids:
                out["source_id"] = cached_source_ids[0]
            out["start_date"] = start
            out["end_date"] = end

        if name == "compare_assets":
            if not self._is_valid_uuid(out.get("source_id")) and cached_source_ids:
                out["source_id"] = cached_source_ids[0]
            if not self._is_valid_uuid(out.get("instrument_a_id")) and cached_instrument_ids:
                out["instrument_a_id"] = cached_instrument_ids[0]
            if not self._is_valid_uuid(out.get("instrument_b_id")) and len(cached_instrument_ids) > 1:
                out["instrument_b_id"] = cached_instrument_ids[1]
            out["start_date"] = start
            out["end_date"] = end

        return out

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
    ) -> None:
        q = question.lower()
        names = [r["name"] for r in runs if r.get("error") is None]

        def call(name: str, args: dict):
            safe_args = self._sanitize_args(name, args, cached_instrument_ids, cached_source_ids)
            raw = execute_tool(name, safe_args, session)
            parsed, err = self._parse_tool_output(raw)
            self._update_caches(name, parsed, cached_instrument_ids, cached_source_ids)
            tool_calls_used.append(name)
            runs.append({"name": name, "args": safe_args, "raw": raw, "data": parsed, "error": err})

        if "list_instruments" not in names:
            call("list_instruments", {})
            names = [r["name"] for r in runs if r.get("error") is None]

        needs_source = any(k in q for k in ["analytics", "average", "trend", "forecast", "risk", "time series", "compare"])
        if needs_source and "list_sources" not in names:
            call("list_sources", {})
            names = [r["name"] for r in runs if r.get("error") is None]

        if any(k in q for k in ["average close", "avg close", "mean close", "analytics"]) and "get_analytics" not in names:
            call("get_analytics", {})

        if any(k in q for k in ["trend", "explain a change"]) and "get_trend" not in names:
            call("get_trend", {})

        if any(k in q for k in ["forecast", "next day"]) and "get_forecast" not in names:
            call("get_forecast", {})

        if any(k in q for k in ["risk", "drawdown", "volatility"]) and "get_risk_signal" not in names:
            call("get_risk_signal", {})

        if "compare" in q and "compare_assets" not in names and len(cached_instrument_ids) >= 2:
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
