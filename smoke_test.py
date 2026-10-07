"""Check both keys work. Run: .venv/bin/python smoke_test.py"""
import os
from dotenv import load_dotenv

load_dotenv()
ok = True

try:
    from elasticsearch import Elasticsearch
    es = Elasticsearch(os.environ["ELASTIC_ENDPOINT"], api_key=os.environ["ELASTIC_API_KEY"])
    print("Elastic OK, version", es.info()["version"]["number"])
except Exception as e:
    ok = False
    print("Elastic FAIL:", e)

try:
    from mistralai.client import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    r = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Say 'ready' in one word."}],
    )
    print("Mistral OK:", r.choices[0].message.content)
except Exception as e:
    ok = False
    print("Mistral FAIL:", e)

raise SystemExit(0 if ok else 1)
