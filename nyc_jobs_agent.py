"""Complain or Get Hired: a Mistral agent over NYC 311 complaints, city job
postings and city payroll, all stored in Elastic.

State a complaint ("rats on my block in Astoria") and Mistral looks up how many
neighbors complained, who owns it, how the city closes it, the open posting that
would let you fix it, and what people in that title really earn. Plain job
questions still work.

Usage:
    .venv/bin/python nyc_jobs_agent.py "rats on my block in Astoria"   # one answer
    .venv/bin/python nyc_jobs_agent.py                                 # chat loop
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
MAX_TOOL_ROUNDS = 5
MAX_TOOL_CALLS = 5

RETURN_FIELDS = [
    "job_id", "business_title", "agency", "salary_range_from", "salary_range_to",
    "salary_frequency", "full_time_part_time_indicator", "career_level",
    "posting_date", "work_location", "civil_service_title",
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
                agency: str | None = None, posted_after: str | None = None,
                title_contains: str | None = None, k: int = 8) -> list[dict]:
    """Hybrid search (semantic + keyword on title/description/skills) plus exact filters."""
    k = max(1, min(int(k or 8), 8))
    filters = _filters(min_salary, full_time, agency, posted_after)
    if title_contains:
        filters.append({"match": {"business_title": title_contains}})
    body = {
        "bool": {
            "should": [
                {"semantic": {"field": "search_text", "query": query}},
                {"multi_match": {"query": query,
                                 "fields": ["business_title^3", "job_description", "preferred_skills"]}},
            ],
            "minimum_should_match": 1,
            "filter": filters,
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
        d["snippet"] = (s.get("job_description") or "")[:600]
        d["how_to_apply"] = (s.get("to_apply") or "")[:300]
        out.append(d)
    return out


def complaints_lookup(es, text: str, borough: str | None = None) -> dict:
    """311 complaints in the last 30 days, via nyc_311_ingest (loaded lazily)."""
    try:
        from nyc_311_ingest import complaints_lookup as _lookup
    except ImportError:
        return {"error": "311 data not loaded yet"}
    return _lookup(es, text, borough)


def real_pay(es, title: str, agency: str | None = None) -> dict:
    """What people in a civil service title really earn, via nyc_payroll_ingest (loaded lazily)."""
    try:
        from nyc_payroll_ingest import real_pay as _pay
    except ImportError:
        return {"error": "payroll data not loaded yet"}
    return _pay(es, title, agency)


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

SYSTEM_PROMPT = """You are "Complain or Get Hired", an assistant over three NYC datasets in Elasticsearch: 311 complaints from the last 30 days, open city job postings (Jobs NYC), and city payroll (what people in each civil service title really earn).
Tagline: stop complaining, start fixing.

General rules:
- Always call a tool before answering. Never answer from memory.
- Never invent a posting, count, agency, salary or pay figure. Only use what the tools returned. If a tool returned nothing or an error, say so plainly.
- Plain English. No markdown tables. Keep it short.

If complaints_lookup returns no_match, do not search jobs; answer in one line that no 311 complaint type matched and ask for one concrete detail, e.g. what is broken and where.

A) When the user describes a problem in the city (rats, noise, potholes, a broken streetlight, heat, trash...), follow this order:
  1. complaints_lookup(text, borough if they named a place; map neighborhoods to their borough, e.g. Astoria -> QUEENS). Call it exactly once.
  2. search_jobs with agency = the lookup's jobs_agency (exact string) and a query describing the work that fixes the complaint (e.g. "pest control inspector rodent exterminator"). If that returns zero postings, call search_jobs again with no agency filter.
  3. real_pay with the chosen posting's civil_service_title (and its agency).
  4. Answer with these sections, each a short line or two, in this order, each heading in bold:
     **Your complaint** - complaint type, how many neighbors complained in the last 30 days, where (borough counts).
     **Who owns it** - the agency.
     **How the city closes it today** - quote one resolution text, shortened.
     **The job that fixes it** - business title, agency, posting salary range with its frequency, one short how-to-apply hint.
     **Real pay in that title** - headcount, average base, average overtime, fiscal year. If real_pay matched a different title than you asked for, say there is no payroll match for that title and do not quote the other title's numbers.
     **Your first month on the job (drafted from the posting)** - exactly three short bullets of work tasks, drawn only from the posting's duties text and the resolution texts. No application steps, no other claims.
     **Stop complaining, start fixing.**
  For follow-ups (e.g. "what about in Brooklyn?"), reuse the earlier complaint and rerun the tools for the new place.

B) For ordinary job questions, behave as a job search assistant:
- Use search_jobs to find postings by meaning; count_jobs for "how many" and "which agencies" questions.
- Put hard requirements (minimum pay, full-time/part-time, agency, date) into the tool filters, not only the query text.
- For each job, cite business title, agency, and salary range with its frequency (Annual, Hourly or Daily).
- If the data cannot answer part of the question, say that part is not in the data.
- Answer with a short list of jobs or numbers, then one line of advice at most.

Tool call caps (hard limits):
- Exactly one complaints_lookup per complaint.
- At most two search_jobs. The second only if the first returned zero postings.
- At most two real_pay. The second only if the first returned matched_how "none"; then try the posting's civil_service_title words without agency.
- At most 5 tool calls in total, then answer.
Each answer section is one or two lines. In "Real pay in that title", if real_pay returned a pay_basis field, state it (e.g. per hour, per annum)."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Hybrid (semantic + keyword) search over NYC city job postings, with optional exact filters. Returns up to k postings with title, civil_service_title, agency, salary, duties snippet and how to apply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the job is about, in plain words, e.g. 'python backend developer' or 'teaching dance to kids'."},
                    "min_salary": {"type": "number", "description": "Minimum top-of-range salary, in the posting's own units (annual dollars for Annual postings)."},
                    "full_time": {"type": "string", "enum": ["F", "P"], "description": "F for full-time, P for part-time."},
                    "agency": {"type": "string", "description": "Exact agency name in upper case, e.g. 'DEPT OF HEALTH/MENTAL HYGIENE'. Use jobs_agency from complaints_lookup, or a name seen in earlier results."},
                    "posted_after": {"type": "string", "description": "ISO date, e.g. '2025-01-01'."},
                    "title_contains": {"type": "string", "description": "Optional words that must appear in the job title."},
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
    {
        "type": "function",
        "function": {
            "name": "complaints_lookup",
            "description": "Look up NYC 311 complaints from the last 30 days matching a described problem. Returns complaint_type, owning agency, jobs_agency (exact agency name for search_jobs), count_30d, counts by borough, and top resolution texts (how the city closed them).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The problem in plain words, e.g. 'rats on my block'."},
                    "borough": {"type": "string", "enum": ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"], "description": "Optional borough to focus on."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "real_pay",
            "description": "What people already in a NYC civil service title really earn, from city payroll: headcount, average base salary, average overtime, average gross.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Civil service title from a posting, e.g. 'PUBLIC HEALTH SANITARIAN'."},
                    "agency": {"type": "string", "description": "Optional agency name to narrow the match."},
                },
                "required": ["title"],
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
    if name == "complaints_lookup":
        return complaints_lookup(es, **args)
    if name == "real_pay":
        return real_pay(es, **args)
    return {"error": f"unknown tool {name}"}


def _should_fall_back(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    return status in (400, 404, 429) or "model" in text or "rate" in text or "capacity" in text


def _cut(t, n: int = 60) -> str:
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _money(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "?"


def _call_line(name: str, args: dict) -> str:
    if name == "complaints_lookup":
        b = f" ({args['borough']})" if args.get("borough") else ""
        return f'-> Elastic 311: "{_cut(args.get("text"))}"{b}'
    if name == "search_jobs":
        a = f"agency {args['agency']}, " if args.get("agency") else ""
        return f'-> Elastic jobs: {a}"{_cut(args.get("query"))}"'
    if name == "real_pay":
        return f"-> Elastic payroll: {_cut(args.get('title'))}"
    if name == "count_jobs":
        q = f' "{_cut(args.get("query"))}"' if args.get("query") else ""
        return f"-> Elastic count:{q}"
    return f"-> Elastic {name}"


def _result_line(name: str, result) -> str:
    if isinstance(result, dict) and "error" in result:
        return f"<- error: {_cut(result['error'])}"
    if name == "complaints_lookup":
        return f"<- {result.get('complaint_type')}, {result.get('count_30d')} complaints, owner {result.get('agency')}"
    if name == "search_jobs":
        if not result:
            return "<- 0 postings"
        return f"<- {len(result)} postings, top: {_cut(result[0].get('business_title'))}"
    if name == "real_pay":
        if not result.get("headcount") or result.get("matched_how") == "none":
            return "<- no match"
        basis = f" ({result['pay_basis']})" if result.get("pay_basis") else ""
        return (f"<- {result.get('headcount')} people, base {_money(result.get('avg_base'))}, "
                f"OT {_money(result.get('avg_overtime'))}{basis}")
    if name == "count_jobs":
        return f"<- {result.get('total')} postings"
    return "<- done"


OTHER_REPLY = "Tell me something broken in the city, like rats or a dark streetlight, and I will find the job that fixes it."
GATE_SYSTEM = ("Label the text. complaint = something wrong in New York City a city agency could fix. "
               "job_question = asking about NYC government jobs, pay or hiring. "
               "other = anything else, including greetings and unrelated statements.")


def _gate(client, text: str, verbose: bool) -> str | None:
    """Label the text with mistral-small. Returns 'complaint', 'job_question', 'other', or None on error."""
    try:
        from typing import Literal
        from pydantic import BaseModel

        class Intent(BaseModel):
            kind: Literal["complaint", "job_question", "other"]

        resp = client.chat.parse(model=FALLBACK_MODEL, response_format=Intent,
                                 messages=[{"role": "system", "content": GATE_SYSTEM},
                                           {"role": "user", "content": text}])
        kind = resp.choices[0].message.parsed.kind
    except Exception:
        return None  # gate failed: skip it
    if verbose:
        print(f"-> Mistral gate: {kind}")
    return kind


def _converse(messages: list, es, client, model: str, verbose: bool) -> str:
    """Run the tool loop on an existing message list (mutated in place). Returns the answer."""
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

    cache: dict = {}
    n_calls = 0
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
            key = (call.function.name, json.dumps(args, sort_keys=True))
            if key in cache:  # identical repeat: reuse, do not hit Elastic again
                result = cache[key]
            else:
                n_calls += 1
                if verbose:
                    print(_call_line(call.function.name, args))
                try:
                    result = _run_tool(es, call.function.name, args)
                except Exception as e:  # report to the model, do not crash the demo
                    result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                cache[key] = result
                if verbose:
                    print(_result_line(call.function.name, result))
            messages.append({"role": "tool", "name": call.function.name,
                             "content": json.dumps(result, default=str), "tool_call_id": call.id})
        if n_calls >= MAX_TOOL_CALLS:
            messages.append({"role": "user", "content": "Answer now, using only the tool results above."})
            resp = complete(tools=TOOLS, tool_choice="none")
            msg = resp.choices[0].message
            break
    else:
        # Used every round on tools; force a final written answer.
        messages.append({"role": "user", "content": "Answer now, using only the tool results above."})
        resp = complete(tools=TOOLS, tool_choice="none")
        msg = resp.choices[0].message

    # Keep only the visible text; drop any reasoning (ThinkChunk) parts.
    answer = msg.content if isinstance(msg.content, str) else "".join(
        c.text for c in (msg.content or []) if type(c).__name__ == "TextChunk")
    messages.append({"role": "assistant", "content": answer})
    if verbose:
        print()
        print(answer)
    return answer


def ask(question: str, es=None, client=None, model: str = DEFAULT_MODEL, verbose: bool = True) -> str:
    """Ask a plain-English question or state a complaint. Mistral picks the Elastic queries, then answers."""
    if not question or not question.strip():
        return "Type a question inside ask(\"...\")."
    es = es or es_client()
    client = client or _client()
    if _gate(client, question, verbose) == "other":
        if verbose:
            print(f"\n{OTHER_REPLY}")
        return OTHER_REPLY
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return _converse(messages, es, client, model, verbose)


def chat(es=None, client=None, model: str = DEFAULT_MODEL, verbose: bool = True) -> None:
    """Chat loop that keeps history across turns. Type quit to exit."""
    es = es or es_client()
    client = client or _client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Complain or Get Hired. Tell me what is wrong on your block, or ask about city jobs. Type quit to exit.")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        if _gate(client, q, verbose) == "other":
            print(f"\n{OTHER_REPLY}\n")
            continue
        mark = len(messages)
        messages.append({"role": "user", "content": q})
        try:
            _converse(messages, es, client, model, verbose)
        except Exception as e:  # keep the loop alive during the demo
            print(f"[error] {type(e).__name__}: {str(e)[:300]}")
            del messages[mark:]  # drop the failed turn so history stays valid
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        chat()
