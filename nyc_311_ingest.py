"""Ingest last-30-day NYC 311 complaint counts (Socrata erm2-nwe9) into Elastic.

Usage:
    .venv/bin/python nyc_311_ingest.py                    # fetch, map agencies, index
    .venv/bin/python nyc_311_ingest.py --check "Rodent"   # print complaints_lookup result
Import:
    from nyc_311_ingest import INDEX_311, AGENCY_MAP_PATH, PERIOD_START, complaints_lookup
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from elasticsearch import ApiError, ConnectionTimeout, helpers
from pydantic import BaseModel

from nyc_jobs_ingest import es_client, EMBED_ID, INDEX as JOBS_INDEX

INDEX_311 = "nyc_311_30d"
AGENCY_MAP_PATH = "agency_map.json"
PERIOD_START = "2026-09-07"
PERIOD_END = "2026-10-07"
URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
TOP_TYPES_WITH_RESOLUTIONS = 60
TIE_MARGIN = 0.0025
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAP_FILE = os.path.join(_HERE, AGENCY_MAP_PATH)
BOROUGHS = ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]


# ---------- fetch ----------
def fetch_rows() -> list[dict]:
    resp = requests.get(URL, params={
        "$select": "agency,agency_name,complaint_type,borough,count(*) as n",
        "$where": f"created_date > '{PERIOD_START}T00:00:00'",
        "$group": "agency,agency_name,complaint_type,borough",
        "$order": "n DESC",
        "$limit": 50000,
    }, timeout=120)
    resp.raise_for_status()
    rows = [r for r in resp.json() if r.get("agency") and r.get("complaint_type")]
    for r in rows:
        r["n"] = int(r["n"])
        r["borough"] = (r.get("borough") or "UNSPECIFIED").upper()
        r["agency_name"] = r.get("agency_name") or r["agency"]
    print(f"Fetched {len(rows)} grouped rows, {sum(r['n'] for r in rows)} complaints")
    return rows


def fetch_resolutions(complaint_type: str) -> list[str]:
    ct = complaint_type.replace("'", "''")
    try:
        resp = requests.get(URL, params={
            "$select": "resolution_description,count(*) as n",
            "$where": f"complaint_type='{ct}' AND created_date > '{PERIOD_START}T00:00:00'",
            "$group": "resolution_description",
            "$order": "n DESC",
            "$limit": 5,
        }, timeout=60)
        resp.raise_for_status()
        return [r["resolution_description"] for r in resp.json() if r.get("resolution_description")][:3]
    except requests.RequestException as e:
        print(f"  resolutions failed for {complaint_type}: {e}")
        return []


# ---------- agency map ----------
class Pair(BaseModel):
    code: str
    jobs_agency: str | None


class Mapping(BaseModel):
    pairs: list[Pair]


def build_agency_map(es, rows: list[dict]) -> dict:
    from dotenv import load_dotenv
    from mistralai.client import Mistral
    load_dotenv(os.path.join(_HERE, ".env"))
    pairs = sorted({(r["agency"], r["agency_name"]) for r in rows})
    codes = sorted({p[0] for p in pairs})
    agg = es.search(index=JOBS_INDEX, size=0, aggs={"a": {"terms": {"field": "agency", "size": 200}}})
    jobs_agencies = [b["key"] for b in agg["aggregations"]["a"]["buckets"]]
    print(f"311 agencies: {len(codes)}, jobs agencies: {len(jobs_agencies)}")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    prompt = (
        "Map each NYC 311 agency code to the single best matching hiring agency string from the "
        "JOBS AGENCIES list. Copy the jobs agency string EXACTLY. Use null if none fits.\n\n"
        "311 AGENCIES (code | name):\n" + "\n".join(f"{c} | {n}" for c, n in pairs) +
        "\n\nJOBS AGENCIES:\n" + "\n".join(jobs_agencies)
    )
    resp = client.chat.parse(
        model="mistral-small-latest",
        messages=[{"role": "system", "content": "You map NYC agency names. Return every 311 code once."},
                  {"role": "user", "content": prompt}],
        response_format=Mapping,
        temperature=0,
    )
    parsed = resp.choices[0].message.parsed
    exact = set(jobs_agencies)
    lower = {a.lower(): a for a in jobs_agencies}
    out = {c: None for c in codes}
    for p in parsed.pairs:
        if p.code not in out:
            continue
        ja = p.jobs_agency
        if ja and ja not in exact:
            ja = lower.get(ja.strip().lower())
        out[p.code] = ja
    with open(_MAP_FILE, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    mapped = sum(1 for v in out.values() if v)
    print(f"Agency map: {mapped}/{len(out)} mapped; null: {[k for k, v in out.items() if not v]}")
    return out


def load_agency_map() -> dict:
    try:
        with open(_MAP_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ---------- index ----------
def ensure_index(es) -> None:
    if es.indices.exists(index=INDEX_311):
        es.indices.delete(index=INDEX_311)
    kw = {"type": "keyword"}
    es.indices.create(index=INDEX_311, mappings={"properties": {
        "agency": kw, "agency_name": kw, "complaint_type": kw, "borough": kw, "jobs_agency": kw,
        "count": {"type": "integer"},
        "period_start": {"type": "date"}, "period_end": {"type": "date"},
        "label": {"type": "semantic_text", "inference_id": EMBED_ID},
        "resolutions": {"type": "text"},
    }})
    print(f"Created index {INDEX_311}")


def build_docs(rows: list[dict], amap: dict, resolutions: dict) -> list[dict]:
    docs = []
    for r in rows:
        docs.append({
            "_id": f"{r['agency']}|{r['complaint_type']}|{r['borough']}",
            "agency": r["agency"], "agency_name": r["agency_name"],
            "complaint_type": r["complaint_type"], "borough": r["borough"],
            "count": r["n"], "jobs_agency": amap.get(r["agency"]),
            "period_start": PERIOD_START, "period_end": PERIOD_END,
            "label": f"{r['complaint_type']} complaint handled by {r['agency_name']}",
            "resolutions": resolutions.get(r["complaint_type"], []),
        })
    return docs


def _bulk(es, docs):
    actions = [{"_index": INDEX_311, "_id": d["_id"],
                "_source": {k: v for k, v in d.items() if k != "_id"}} for d in docs]
    return helpers.bulk(es.options(request_timeout=120), actions, raise_on_error=False,
                        raise_on_exception=False, chunk_size=len(actions))


def load_docs(es, docs: list[dict], chunk_size: int = 50) -> tuple[int, int]:
    indexed, errors, t0 = 0, 0, time.time()
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i + chunk_size]
        try:
            ok, errs = _bulk(es, chunk)
        except (ConnectionTimeout, ApiError) as e:
            ok, errs = 0, chunk
            print(f"  chunk raised: {str(e)[:200]}")
        if errs:
            failed = {list(e.values())[0].get("_id") for e in errs if isinstance(e, dict) and len(e) == 1} \
                if isinstance(errs[0], dict) and "_id" not in errs[0] else {d["_id"] for d in errs}
            retry = [d for d in chunk if d["_id"] in failed] or chunk
            print(f"  chunk {i // chunk_size + 1}: {len(errs)} failed, retrying in 10s")
            time.sleep(10)
            try:
                ok2, errs = _bulk(es, retry)
            except (ConnectionTimeout, ApiError) as e:
                ok2, errs = 0, retry
            ok += ok2
        indexed += ok
        errors += len(errs)
        if (i // chunk_size) % 10 == 0:
            print(f"  indexed {indexed}/{len(docs)}, errors {errors}, {time.time() - t0:.0f}s", flush=True)
    es.indices.refresh(index=INDEX_311)
    return indexed, errors


# ---------- lookup ----------
def _norm_borough(b: str | None) -> str | None:
    if not b:
        return None
    b = b.strip().upper()
    aliases = {"NYC": None, "NEW YORK": "MANHATTAN", "THE BRONX": "BRONX", "STATEN": "STATEN ISLAND",
               "SI": "STATEN ISLAND", "BK": "BROOKLYN", "BKLYN": "BROOKLYN",
               "ASTORIA": "QUEENS", "FLUSHING": "QUEENS", "JAMAICA": "QUEENS", "HARLEM": "MANHATTAN",
               "WILLIAMSBURG": "BROOKLYN", "BUSHWICK": "BROOKLYN", "BED-STUY": "BROOKLYN"}
    if b in aliases:
        return aliases[b]
    for full in BOROUGHS:
        if b == full or b in full or full in b:
            return full
    return None  # unknown (e.g. a neighborhood like "Astoria"): fall back to citywide


def complaints_lookup(es, text: str, borough: str | None = None) -> dict:
    """Free text like 'rats on my block' -> the best matching complaint type and its numbers.
    Returns {"complaint_type", "agency", "agency_name", "jobs_agency", "count_30d", "by_borough",
             "top_resolutions", "period_start", "period_end"}."""
    borough = _norm_borough(borough)
    hits = es.search(index=INDEX_311, size=50,
                     query={"bool": {"must": [{"semantic": {"field": "label", "query": text}}]}},
                     source_excludes=["label"])["hits"]["hits"]
    empty = {"complaint_type": None, "agency": None, "agency_name": None, "jobs_agency": None,
             "count_30d": 0, "by_borough": [], "top_resolutions": [],
             "period_start": PERIOD_START, "period_end": PERIOD_END}
    if not hits:
        return empty
    top = hits[0]["_source"]
    # Near-ties (labels differ by a word, e.g. "Noise - Park" vs "Noise - Residential"):
    # among types scoring within TIE_MARGIN of the best, pick the most-reported one citywide.
    best = hits[0]["_score"]
    cands = []
    for h in hits:
        k = (h["_source"]["complaint_type"], h["_source"]["agency"])
        if h["_score"] >= best - TIE_MARGIN and k not in cands:
            cands.append(k)
    if len(cands) > 1:
        tot = es.search(index=INDEX_311, size=0,
                        query={"terms": {"complaint_type": [c for c, _ in cands]}},
                        aggs={"k": {"multi_terms": {"terms": [{"field": "complaint_type"}, {"field": "agency"}],
                                                    "size": 50},
                                    "aggs": {"c": {"sum": {"field": "count"}}}}})
        sums = {tuple(b["key"]): b["c"]["value"] for b in tot["aggregations"]["k"]["buckets"]}
        top = {"complaint_type": None, "agency": None}
        top["complaint_type"], top["agency"] = max(cands, key=lambda k: sums.get(k, 0))
    ctype, agency = top["complaint_type"], top["agency"]
    flt = [{"term": {"complaint_type": ctype}}, {"term": {"agency": agency}}]
    res = es.search(index=INDEX_311, size=1, source_excludes=["label"],
                    query={"bool": {"filter": flt}}, sort=[{"count": "desc"}],
                    aggs={"total": {"sum": {"field": "count"}},
                          "boro": {"terms": {"field": "borough", "size": 10, "order": {"c": "desc"}},
                                   "aggs": {"c": {"sum": {"field": "count"}}}}})
    src = res["hits"]["hits"][0]["_source"] if res["hits"]["hits"] else top
    by_boro = [{"borough": b["key"], "count": int(b["c"]["value"])}
               for b in res["aggregations"]["boro"]["buckets"]]
    total = int(res["aggregations"]["total"]["value"])
    if borough:
        total = next((b["count"] for b in by_boro if b["borough"] == borough), 0)
    return {
        "complaint_type": ctype, "agency": agency, "agency_name": src.get("agency_name"),
        "jobs_agency": src.get("jobs_agency"), "count_30d": total, "by_borough": by_boro,
        "top_resolutions": (src.get("resolutions") or [])[:3],
        "period_start": PERIOD_START, "period_end": PERIOD_END,
    }


# ---------- main ----------
def main() -> None:
    t0 = time.time()
    es = es_client()
    rows = fetch_rows()
    amap = build_agency_map(es, rows)
    totals: dict[str, int] = {}
    for r in rows:
        totals[r["complaint_type"]] = totals.get(r["complaint_type"], 0) + r["n"]
    top_types = sorted(totals, key=totals.get, reverse=True)[:TOP_TYPES_WITH_RESOLUTIONS]
    with ThreadPoolExecutor(8) as ex:
        resolutions = dict(zip(top_types, ex.map(fetch_resolutions, top_types)))
    print(f"Resolutions fetched for {sum(1 for v in resolutions.values() if v)}/{len(top_types)} types")
    ensure_index(es)
    docs = build_docs(rows, amap, resolutions)
    indexed, errors = load_docs(es, docs)
    count = es.count(index=INDEX_311)["count"]
    print(f"Done: rows {len(rows)}, indexed {indexed}, errors {errors}, es.count {count}, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=str, default=None)
    ap.add_argument("--borough", type=str, default=None)
    a = ap.parse_args()
    if a.check:
        print(json.dumps(complaints_lookup(es_client(), a.check, a.borough), indent=2))
    else:
        main()
