"""Mistral agent over NYC city government job postings stored in Elastic.

Mistral reads the question, picks which Elastic query to run (search_jobs or
count_jobs, with filters), reads the results, and writes the answer.

Usage:
    .venv/bin/python nyc_jobs_agent.py "city software jobs paying over 100k"
"""
import json
import os
import sys

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError

from nyc_jobs_ingest import INDEX, es_client

DEFAULT_MODEL = "mistral-large-4"
FALLBACK_MODEL = "mistral-small-latest"
MAX_TOOL_ROUNDS = 4

RETURN_FIELDS = [
    "job_id", "business_title", "agency", "salary_range_from", "salary_range_to",
    "salary_frequency", "full_time_part_time_indicator", "career_level",
    "posting_date", "work_location",
]


# ---------------------------------------------------------------- Elastic side

def _filters(min_salary=None, full_time=None, agency=None, posted_after=None) -> list[dict]:
    f = []
    if min_salary is not None:
        f.append({"range": {"salary_range_to": {"gte": float(min_salary)}}})
    if full_time:
        f.append({"term": {"full_time_part_time_indicator": str(full_time).upper()[:1]}})
    if agency:
        f.append({"term": {"agency": agency}})
    if posted_after:
        f.append({"range": {"posting_date": {"gte": posted_after}}})
    return f


def search_jobs(es, query: str, min_salary: float | None = None, full_time: str | None = None,
                agency: str | None = None, posted_after: str | None = None, k: int = 8) -> list[dict]:
    """Semantic search on search_text plus exact filters. Returns trimmed postings."""
    k = max(1, min(int(k or 8), 8))
    body = {
        "bool": {
            "must": [{"semantic": {"field": "search_text", "query": query}}],
            "filter": _filters(min_salary, full_time, agency, posted_after),
        }
    }
    resp = es.search(
        index=INDEX, query=body, size=k,
        source=RETURN_FIELDS + ["job_description", "to_apply"],
    )
    out = []
    for h in resp["hits"]["hits"]:
        s = h["_source"]
        d = {f: s.get(f) for f in RETURN_FIELDS}
        d["snippet"] = (s.get("job_description") or "")[:400]
        d["how_to_apply"] = (s.get("to_apply") or "")[:300]
        out.append(d)
    return out


def count_jobs(es, query: str | None = None, min_salary=None, full_time=None) -> dict:
    """Counts and aggregations. Semantic match only if a query is given."""
    must = [{"semantic": {"field": "search_text", "query": query}}] if query else [{"match_all": {}}]
    body = {"bool": {"must": must, "filter": _filters(min_salary, full_time)}}
    resp = es.search(
        index=INDEX, query=body, size=0, track_total_hits=True,
        aggs={
            "by_agency": {"terms": {"field": "agency", "size": 10}},
            "salary": {"stats": {"field": "salary_range_to"}},
            "by_career_level": {"terms": {"field": "career_level", "size": 10}},
        },
    )
    aggs = resp["aggregations"]
    salary = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in aggs["salary"].items()}
    return {
        "total": resp["hits"]["total"]["value"],
        "by_agency": [{"agency": b["key"], "count": b["doc_count"]} for b in aggs["by_agency"]["buckets"]],
        "salary": salary,
        "by_career_level": [{"career_level": b["key"], "count": b["doc_count"]}
                            for b in aggs["by_career_level"]["buckets"]],
    }


# ---------------------------------------------------------------- Mistral side

SYSTEM_PROMPT = """You are an assistant over NYC city government job postings (the Jobs NYC Postings dataset) stored in Elasticsearch.
Rules:
- Always call a tool before answering. Never answer from memory.
- Use search_jobs to find specific postings by meaning. Use count_jobs for "how many" and "which agencies" questions.
- Put hard requirements (minimum pay, full-time/part-time, agency, date) into the tool filters, not only the query text.
- For each job you mention, cite: business title, agency, and salary range with its frequency (Annual, Hourly or Daily).
- Never invent a posting, salary, or agency. Only use what the tools returned. If nothing fits, say so plainly.
- Do not add claims about typical pay, schedules, or benefits unless a returned posting says so.
- Say counts plainly, as numbers.
- If the data cannot answer part of the question (for example, end time of a shift), say that part is not in the data.
- Make at most 3 tool calls in total, then answer. Prefer one well-filtered call.
- Keep the answer short: a short list of jobs or numbers, then one line of advice at most."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Semantic search over NYC city job postings in Elasticsearch, with optional exact filters. Returns up to k postings with title, agency, salary, and a snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the job is about, in plain words, e.g. 'python backend developer' or 'teaching dance to kids'."},
                    "min_salary": {"type": "number", "description": "Minimum top-of-range salary, in the posting's own units (annual dollars for Annual postings)."},
                    "full_time": {"type": "string", "enum": ["F", "P"], "description": "F for full-time, P for part-time."},
                    "agency": {"type": "string", "description": "Exact agency name in upper case, e.g. 'DEPT OF ENVIRONMENT PROTECTION'. Only use a name seen in earlier results."},
                    "posted_after": {"type": "string", "description": "ISO date, e.g. '2025-01-01'."},
                    "k": {"type": "integer", "description": "Number of postings to return, 1 to 8.", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_jobs",
            "description": "Count NYC city job postings matching optional filters. Returns the total, top 10 agencies by count, salary stats on salary_range_to, and counts by career level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional topic to restrict the count by meaning. Omit to count all postings."},
                    "min_salary": {"type": "number", "description": "Minimum top-of-range salary."},
                    "full_time": {"type": "string", "enum": ["F", "P"], "description": "F for full-time, P for part-time."},
                },
            },
        },
    },
]


def _client() -> Mistral:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _run_tool(es, name: str, args: dict):
    if name == "search_jobs":
        return search_jobs(es, **args)
    if name == "count_jobs":
        return count_jobs(es, **args)
    return {"error": f"unknown tool {name}"}


def _should_fall_back(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    return status in (400, 404, 429) or "model" in text or "rate" in text or "capacity" in text


def ask(question: str, es=None, client=None, model: str = DEFAULT_MODEL, verbose: bool = True) -> str:
    """Ask a plain-English question. Mistral picks the Elastic queries, then answers."""
    if not question or not question.strip():
        return "Type a question inside ask(\"...\")."
    es = es or es_client()
    client = client or _client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    state = {"model": model, "fell_back": False}

    def _chat(**kw):
        # mistral-large-4 is a reasoning model; reasoning off keeps the demo fast and the output clean.
        extra = {"reasoning_effort": "none"} if "large-4" in state["model"] else {}
        return client.chat.complete(model=state["model"], messages=messages, **extra, **kw)

    def complete(**kw):
        try:
            return _chat(**kw)
        except MistralError as e:
            if state["fell_back"] or not _should_fall_back(e):
                raise
            print(f"[fallback] {state['model']} failed ({getattr(e, 'status_code', '?')}); retrying with {FALLBACK_MODEL}")
            state["model"], state["fell_back"] = FALLBACK_MODEL, True
            return _chat(**kw)

    if verbose:
        print(f"Q: {question}\n[model] {model}")

    for round_no in range(1, MAX_TOOL_ROUNDS + 1):
        # First round must use a tool; later rounds may answer.
        resp = complete(tools=TOOLS, tool_choice="any" if round_no == 1 else "auto")
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            break
        messages.append(msg)
        for call in calls:
            args = call.function.arguments
            args = json.loads(args) if isinstance(args, str) else dict(args or {})
            if verbose:
                print(f"[Mistral -> Elastic] {call.function.name}({json.dumps(args)})")
            try:
                result = _run_tool(es, call.function.name, args)
            except Exception as e:  # report to the model, do not crash the demo
                result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            if verbose:
                n = len(result) if isinstance(result, list) else result.get("total", result.get("error"))
                print(f"[Elastic -> Mistral] {n} {'postings' if isinstance(result, list) else ''}".rstrip())
            messages.append({"role": "tool", "name": call.function.name,
                             "content": json.dumps(result, default=str), "tool_call_id": call.id})
    else:
        # Used every round on tools; force a final written answer.
        messages.append({"role": "user", "content": "Answer now, using only the tool results above."})
        resp = complete(tools=TOOLS, tool_choice="none")
        msg = resp.choices[0].message

    # Keep only the visible text; drop any reasoning (ThinkChunk) parts.
    answer = msg.content if isinstance(msg.content, str) else "".join(
        c.text for c in (msg.content or []) if type(c).__name__ == "TextChunk")
    if verbose:
        print(f"[answered by] {state['model']}\n")
        print(answer)
    return answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "city software jobs paying over 100k"
    ask(q)
