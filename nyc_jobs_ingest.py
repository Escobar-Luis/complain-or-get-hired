"""Ingest NYC Jobs postings (Socrata kpav-sd4t) into Elastic with Mistral semantic search.

Usage:
    .venv/bin/python nyc_jobs_ingest.py --limit 1200   # newest 1200 External postings
    .venv/bin/python nyc_jobs_ingest.py                # all External postings
"""
import argparse
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from elasticsearch import ApiError, ConnectionTimeout, NotFoundError

INDEX = "nyc_jobs"
EMBED_ID = "mistral-embeddings"      # text_embedding, mistral-embed
CHAT_ID = "mistral-chat"             # chat_completion, mistral-large-4
DATASET = "kpav-sd4t"

CHAT_MODEL = "mistral-large-4"
CHAT_MODEL_FALLBACK = "mistral-large-latest"
SEARCH_TEXT_MAX = 6000

KEYWORD_FIELDS = [
    "job_id", "agency", "posting_type", "civil_service_title", "job_category",
    "full_time_part_time_indicator", "career_level", "salary_frequency", "level",
]
TEXT_FIELDS = [
    "job_description", "minimum_qual_requirements", "preferred_skills", "to_apply",
    "work_location", "residency_requirement",
]
FLOAT_FIELDS = ["salary_range_from", "salary_range_to", "number_of_positions"]
DATE_FIELDS = ["posting_date", "post_until", "posting_updated"]


def es_client() -> Elasticsearch:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    es = Elasticsearch(
        os.environ["ELASTIC_ENDPOINT"],
        api_key=os.environ["ELASTIC_API_KEY"],
        request_timeout=120,
    )
    es.info()
    return es


def _endpoint_exists(es, inference_id: str) -> bool:
    try:
        es.inference.get(inference_id=inference_id)
        return True
    except NotFoundError:
        return False


def _put_endpoint(es, task_type: str, inference_id: str, model: str) -> None:
    es.inference.put(
        task_type=task_type,
        inference_id=inference_id,
        inference_config={
            "service": "mistral",
            "service_settings": {"api_key": os.environ["MISTRAL_API_KEY"], "model": model},
        },
    )


def ensure_endpoints(es) -> None:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if _endpoint_exists(es, EMBED_ID):
        print(f"Endpoint {EMBED_ID} exists")
    else:
        _put_endpoint(es, "text_embedding", EMBED_ID, "mistral-embed")
        print(f"Created endpoint {EMBED_ID} (mistral-embed)")
    if _endpoint_exists(es, CHAT_ID):
        print(f"Endpoint {CHAT_ID} exists")
    else:
        try:
            _put_endpoint(es, "chat_completion", CHAT_ID, CHAT_MODEL)
            print(f"Created endpoint {CHAT_ID} ({CHAT_MODEL})")
        except ApiError as e:
            print(f"{CHAT_MODEL} rejected ({e.status_code}); falling back to {CHAT_MODEL_FALLBACK}")
            _put_endpoint(es, "chat_completion", CHAT_ID, CHAT_MODEL_FALLBACK)
            print(f"Created endpoint {CHAT_ID} ({CHAT_MODEL_FALLBACK})")


def ensure_index(es, recreate: bool = True) -> None:
    if es.indices.exists(index=INDEX):
        if not recreate:
            print(f"Index {INDEX} exists, keeping it")
            return
        es.indices.delete(index=INDEX)
    props = {f: {"type": "keyword"} for f in KEYWORD_FIELDS}
    props.update({f: {"type": "text"} for f in TEXT_FIELDS})
    props.update({f: {"type": "float"} for f in FLOAT_FIELDS})
    props.update({f: {"type": "date", "format": "strict_date_optional_time||epoch_millis"} for f in DATE_FIELDS})
    props["business_title"] = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
    props["search_text"] = {"type": "semantic_text", "inference_id": EMBED_ID}
    es.indices.create(index=INDEX, mappings={"properties": props})
    print(f"Created index {INDEX}")


def fetch_jobs(limit: int = 5000) -> list[dict]:
    url = f"https://data.cityofnewyork.us/resource/{DATASET}.json"
    rows, offset, page = [], 0, 1000
    # Over-fetch is unnecessary: the $where already filters to External postings.
    while len(rows) < limit:
        resp = requests.get(url, params={
            "$where": "posting_type='External'",
            "$order": "posting_date DESC, job_id",
            "$limit": min(page, limit - len(rows)),
            "$offset": offset,
        }, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < page:
            break
    seen, out = set(), []
    for r in rows:
        jid = r.get("job_id")
        if jid and jid not in seen:
            seen.add(jid)
            out.append(r)
    print(f"Fetched {len(rows)} External rows, {len(out)} unique job_ids")
    return out


def _iso_date(v: str) -> str | None:
    # Socrata mixes ISO ("2026-10-01T00:00:00.000") and "05-DEC-2026".
    if len(v) >= 10 and v[4] == "-" and v[7] == "-":
        return v
    for fmt in ("%d-%b-%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(v.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _clean(r: dict) -> dict:
    doc = {}
    for k, v in r.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if k in FLOAT_FIELDS:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        if k in DATE_FIELDS:
            v = _iso_date(v)
            if v is None:
                continue
        doc[k] = v
    parts = [doc.get("business_title", ""), doc.get("agency", ""),
             doc.get("job_description", ""), doc.get("preferred_skills", "")]
    doc["search_text"] = " | ".join(parts)[:SEARCH_TEXT_MAX]
    return doc


def _bulk_chunk(es, docs: list[dict]) -> tuple[int, list]:
    actions = [{"_index": INDEX, "_id": d["job_id"], "_source": d} for d in docs]
    ok, errs = helpers.bulk(es.options(request_timeout=120), actions, raise_on_error=False,
                            raise_on_exception=False, chunk_size=len(actions))
    return ok, errs


def load_jobs(es, rows: list[dict], chunk_size: int = 20) -> tuple[int, int]:
    docs = [_clean(r) for r in rows]
    indexed, errors, t0 = 0, 0, time.time()
    n_chunks = (len(docs) + chunk_size - 1) // chunk_size
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i:i + chunk_size]
        try:
            ok, errs = _bulk_chunk(es, chunk)
        except (ConnectionTimeout, ApiError) as e:
            ok, errs = 0, [{"index": {"_id": d["job_id"], "status": 429, "error": str(e)}} for d in chunk]
        if errs:
            # Retry the failed docs once after a pause (covers Mistral 429s and timeouts).
            first = list(errs[0].values())[0]
            retryable = [e for e in errs if list(e.values())[0].get("status") in (429, 500, 502, 503, 504, 408)]
            if not retryable:
                print(f"  chunk {i // chunk_size + 1}: {len(errs)} failed, not retryable: {str(first)[:300]}")
                indexed += ok
                errors += len(errs)
                continue
            failed_ids = {list(e.values())[0].get("_id") for e in retryable}
            ok_hold = len(errs) - len(retryable)
            retry = [d for d in chunk if d["job_id"] in failed_ids]
            first = list(errs[0].values())[0]
            print(f"  chunk {i // chunk_size + 1}: {len(errs)} failed "
                  f"(status {first.get('status')}), retrying in 10s")
            time.sleep(10)
            try:
                ok2, errs = _bulk_chunk(es, retry)
            except (ConnectionTimeout, ApiError) as e:
                ok2, errs = 0, retry
                print(f"  retry raised: {e}")
            ok += ok2
            errs = list(errs) + [None] * ok_hold
            if errs and errs[0] is not None:
                first = errs[0] if isinstance(errs[0], dict) else {}
                print(f"  retry still failed {len(errs)}: {str(first)[:300]}")
        indexed += ok
        errors += len(errs)
        print(f"chunk {i // chunk_size + 1}/{n_chunks}: indexed {indexed}, errors {errors}, "
              f"{time.time() - t0:.0f}s", flush=True)
    es.indices.refresh(index=INDEX)
    return indexed, errors


def main(limit: int | None = None) -> None:
    t0 = time.time()
    es = es_client()
    print(f"Connected to Elasticsearch {es.info()['version']['number']}")
    ensure_endpoints(es)
    ensure_index(es, recreate=True)
    rows = fetch_jobs(limit or 5000)
    indexed, errors = load_jobs(es, rows)
    count = es.count(index=INDEX)["count"]
    print(f"Done: fetched {len(rows)}, indexed {indexed}, errors {errors}, "
          f"es.count {count}, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
