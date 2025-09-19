from __future__ import annotations
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
from aiohttp import ClientTimeout

from agentflow.tools.search.backend.base import SearchBackend
from agentflow.tools.base import BaseTool, ToolCallRequest, ToolCallResult
from agentflow.agent.summary.interface import SummarizerInterface

def _run_coroutine_blocking(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import threading

    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, BaseException] = {}

    def _target():
        try:
            result_holder["v"] = asyncio.run(coro)
        except BaseException as e:
            exc_holder["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if "e" in exc_holder:
        raise exc_holder["e"]
    return result_holder.get("v")

class AsyncSearchTool(BaseTool):
    """Async search tool which can switch from search backends.
    """
    name = "search"
    description = "Async search tool with pluggable backends (blocking API)."

    def __init__(
        self,
        backend: SearchBackend,
        *,
        max_rounds = 3,
        timeout: int = 60,
        top_k: int = 3,
        concurrent_limit: int = 2,
        trust_env: bool = False,
        proxy: Optional[str] = None,
        fetch_details: bool = True,
        detail_concurrency: int = 3,
        max_length: int = 10000,
        enable_summarize: bool = False,
        summarize_engine: SummarizerInterface = None,  
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config,max_rounds=max_rounds)
        self.backend = backend
        self.timeout = timeout
        self.top_k = top_k
        self.concurrent_limit = concurrent_limit
        self.trust_env = trust_env
        self.proxy = proxy
        self.fetch_details = fetch_details
        self.detail_concurrency = detail_concurrency
        self.max_length = max_length
        self.enable_summarize = enable_summarize
        self.summarize_engine = summarize_engine

    def run_one(self, call: ToolCallRequest, **kwargs: Any) -> ToolCallResult:
        if self._is_quota_exceeded(call):
            return self._make_exceeded_result(call)

        query = str(call.content)
        results = _run_coroutine_blocking(self._run_batch_async([query]))
        text = results[0]["formatted"]
        meta = results[0]["meta"]
        meta.update(getattr(call, "meta", {}) or {})
        final_content = text

        if self.enable_summarize and self.summarize_engine:
            summary, summarize_meta = self.summarize_engine.summarize(text, meta)
            meta.update(summarize_meta or {})
            final_content = summary

        return ToolCallResult(
            tool_name=self.name,
            request_content=query,
            output=final_content,
            meta=meta,
            error=None,
            index=call.index,
            call=call,
        )

    def run_batch(self, calls: List[ToolCallRequest], **kwargs: Any) -> List[ToolCallResult]:
        def _runner(allowed_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
            if not allowed_calls:
                return []
            queries = [str(c.content) for c in allowed_calls]
            results = _run_coroutine_blocking(self._run_batch_async(queries))
            outs = [r["formatted"] for r in results]
            metas = [r["meta"] for r in results]
            for m, c in zip(metas, allowed_calls):
                m.update(getattr(c, "meta", {}) or {})

            if self.enable_summarize and self.summarize_engine:
                summaries, summarize_metas = self.summarize_engine.summarize_batch(outs, metas)
                outs = summaries
                for m, sm in zip(metas, summarize_metas):
                    m.update(sm or {})

            packed: List[ToolCallResult] = []
            for c, out, m in zip(allowed_calls, outs, metas):
                packed.append(
                    ToolCallResult(
                        tool_name=self.name,
                        request_content=c.content,
                        output=out,
                        meta=m,
                        error=None,
                        index=c.index,
                        call=c,
                    )
                )
            return packed

        return self._apply_round_quota(calls, _runner)

    async def _run_batch_async(self, queries: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        timeout = ClientTimeout(total=self.timeout)

        connector = aiohttp.TCPConnector(limit_per_host=self.concurrent_limit, ssl=False)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=self.trust_env, connector=connector) as session:
            # 1) search in parallel
            hits_lists = await asyncio.gather(
                *[self.backend.search(session, q, top_k=self.top_k) for q in queries],
                return_exceptions=True,
            )
            # 2) details (optional)
            if self.fetch_details:
                lengths: List[int] = []
                flattened: List[Dict[str, Any]] = []
                for hits in hits_lists:
                    if isinstance(hits, Exception) or not hits:
                        lengths.append(0)
                    else:
                        lengths.append(len(hits))
                        flattened.extend(hits)

                if flattened:
                    try:
                        enriched_flat = await self.backend.fetch_details(
                            session,
                            flattened,
                            max_length=self.max_length,
                            concurrency=self.detail_concurrency,  
                            proxy=self.proxy,
                        )
                        enriched_lists: List[List[Dict[str, Any]]] = []
                        offset = 0
                        for L in lengths:
                            if L == 0:
                                enriched_lists.append([])
                            else:
                                enriched_lists.append(enriched_flat[offset: offset + L])
                                offset += L
                        hits_lists = enriched_lists
                    except Exception:

                        enriched_lists: List[List[Dict[str, Any]]] = []
                        for hits in hits_lists:
                            if isinstance(hits, Exception) or not hits:
                                enriched_lists.append([])
                            else:
                                enriched = await self.backend.fetch_details(
                                    session,
                                    hits,
                                    max_length=self.max_length,
                                    concurrency=self.detail_concurrency,
                                    proxy=self.proxy,
                                )
                                enriched_lists.append(enriched)
                        hits_lists = enriched_lists

            # 3) format each result
            for q, hits in zip(queries, hits_lists):
                if isinstance(hits, Exception) or not hits:
                    out.append({"formatted": "Search Failed", "meta": {"raw_hits": [], "query": q}})
                    continue
                formatted = self._format_hits(hits, q)
                out.append({"formatted": formatted, "meta": {"raw_hits": hits, "query": q}})
        return out

    def _format_hits(self, hits: List[Dict[str, Any]], query: str) -> str:
        if not hits:
            return "Search Failed"
        parts: List[str] = []
        share = max(1, self.max_length // max(1, len(hits)))
        for idx, h in enumerate(hits):
            title = (h.get("title") or "").strip() or "(no title)"
            url = h.get("url") or ""
            body = (h.get("content") or h.get("snippet") or "").replace("\n", " ")
            parts.append(f"[{idx}] Title: {title} Source: {url}\n{body[:share]}\n")
        text = "\n".join(parts)[: self.max_length]
        return text