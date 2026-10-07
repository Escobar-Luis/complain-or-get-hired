"""Ingest NYC Citywide Payroll (Socrata k397-673e) aggregates into Elastic.

One doc per (civil service title, agency): headcount and average base, overtime, gross pay.
Lets the agent say what people already in a title really earn.

Usage:
    .venv/bin/python nyc_payroll_ingest.py                    # load FY2025
    .venv/bin/python nyc_payroll_ingest.py --check "TEACHER"  # print real_pay
"""
import argparse
import json
import time

import requests
from elasticsearch import helpers

from nyc_jobs_ingest import es_client, INDEX as JOBS_INDEX  # noqa: F401

INDEX_PAY = "nyc_payroll_fy2025"
DATASET = "k397-673e"
FISCAL_YEAR = 2025
URL = f"https://data.cityofnewyork.us/resource/{DATASET}.json"

MAPPINGS = {
    "properties": {
        "title": {"type": "keyword", "fields": {"text": {"type": "text"}}},
        "agency": {"type": "keyword", "fields": {"text": {"type": "text"}}},
        "headcount": {"type": "integer"},
        "avg_base": {"type": "float"},
        "avg_overtime": {"type": "float"},
        "avg_gross": {"type": "float"},
        "fiscal_year": {"type": "integer"},
    }
}


def fetch(year: int) -> list[dict]:
    params = {
        "$select": "title_description,agency_name,avg(base_salary) as base,"
                   "avg(total_ot_paid) as ot,avg(regular_gross_paid) as gross,count(*) as n",
        "$where": f"fiscal_year={year} AND pay_basis='per Annum'",
        "$group": "title_description,agency_name",
        "$order": "n DESC",
        "$limit": 50000,
    }
    r = requests.get(URL, params=params, timeout=180)
    r.raise_for_status()
    return r.json()


def _f(v) -> float:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return 0.0


def to_docs(rows: list[dict], year: int):
    for row in rows:
        title = (row.get("title_description") or "").strip().upper()
        agency = (row.get("agency_name") or "").strip().upper()
        if not title:
            continue
        yield {
            "_index": INDEX_PAY,
            "_id": f"{title}|{agency}",
            "_source": {
                "title": title,
                "agency": agency,
                "headcount": int(float(row.get("n") or 0)),
                "avg_base": _f(row.get("base")),
                "avg_overtime": _f(row.get("ot")),
                "avg_gross": _f(row.get("gross")),
                "fiscal_year": year,
            },
        }


def load() -> None:
    t0 = time.time()
    year = FISCAL_YEAR
    rows = fetch(year)
    if not rows:
        year = 2024
        print("FY2025 returned nothing, falling back to FY2024")
        rows = fetch(year)
    print(f"Fetched {len(rows)} grouped rows for FY{year} in {time.time() - t0:.1f}s")
    es = es_client()
    if not es.indices.exists(index=INDEX_PAY):
        es.indices.create(index=INDEX_PAY, mappings=MAPPINGS)
        print(f"Created index {INDEX_PAY}")
    ok, errors = helpers.bulk(es, to_docs(rows, year), chunk_size=500, raise_on_error=False)
    es.indices.refresh(index=INDEX_PAY)
    print(f"Indexed {ok} docs, {len(errors)} errors, total {time.time() - t0:.1f}s")
    if errors:
        print("First error:", errors[0])


def _empty(title: str) -> dict:
    return {"title": title, "fiscal_year": FISCAL_YEAR, "headcount": 0, "avg_base": 0.0,
            "avg_overtime": 0.0, "avg_gross": 0.0, "agencies": [], "matched_how": "none"}


def _search(es, query: dict, size: int = 500) -> list[dict]:
    res = es.search(index=INDEX_PAY, query=query, size=size)
    return [h["_source"] for h in res["hits"]["hits"]]


def real_pay(es, title: str, agency: str | None = None) -> dict:
    """Civil service title (any case) -> what people in it really earn.
    Returns {"title", "fiscal_year", "headcount", "avg_base", "avg_overtime", "avg_gross",
             "agencies": [{"agency", "headcount", "avg_base", "avg_overtime"}, ...up to 5],
             "matched_how": "exact" | "fuzzy" | "none"}"""
    t = (title or "").strip().upper()
    if not t:
        return _empty(t)
    agency_filter = [{"match": {"agency.text": agency}}] if agency else []

    matched_how = "exact"
    docs = _search(es, {"bool": {"filter": [{"term": {"title": t}}], "must": agency_filter}})
    if not docs:
        top = _search(es, {"bool": {"must": [{"match": {"title.text": {
            "query": t, "operator": "and", "fuzziness": "AUTO"}}}] + agency_filter}}, size=1)
        if not top:
            top = _search(es, {"bool": {"must": [{"match": {"title.text": t}}] + agency_filter}}, size=1)
        if not top:
            return _empty(t)
        t = top[0]["title"]
        matched_how = "fuzzy"
        docs = _search(es, {"bool": {"filter": [{"term": {"title": t}}], "must": agency_filter}})
    if not docs:
        return _empty(t)

    n = sum(d["headcount"] for d in docs) or 1

    def wavg(field: str) -> float:
        return round(sum(d[field] * d["headcount"] for d in docs) / n, 2)

    top5 = sorted(docs, key=lambda d: d["headcount"], reverse=True)[:5]
    return {
        "title": t,
        "fiscal_year": docs[0].get("fiscal_year", FISCAL_YEAR),
        "headcount": sum(d["headcount"] for d in docs),
        "avg_base": wavg("avg_base"),
        "avg_overtime": wavg("avg_overtime"),
        "avg_gross": wavg("avg_gross"),
        "agencies": [{"agency": d["agency"], "headcount": d["headcount"],
                      "avg_base": d["avg_base"], "avg_overtime": d["avg_overtime"]} for d in top5],
        "matched_how": matched_how,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="civil service title to look up")
    ap.add_argument("--agency", help="optional agency filter for --check")
    args = ap.parse_args()
    if args.check:
        es = es_client()
        print(json.dumps(real_pay(es, args.check, args.agency), indent=2))
    else:
        load()


if __name__ == "__main__":
    main()
