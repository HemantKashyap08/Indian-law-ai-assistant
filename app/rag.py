import json
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

import chromadb
from google import genai
from google.genai import types

client = genai.Client()  # reads GEMINI_API_KEY from the environment
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBED_MODEL = "gemini-embedding-001"

coll = chromadb.PersistentClient(path=str(BASE / "chroma_db")).get_collection("indian_laws")
PROMPT = (BASE / "prompts" / "nyayamitra.txt").read_text(encoding="utf-8")

NEW_CODES = ["BNS", "BNSS", "BSA"]
OLD_CODES = ["IPC", "CRPC", "IEA"]
CUTOFF = date(2024, 7, 1)

EXTRACT_PROMPT = """Read the incident and return JSON only:
{"incident_date": "YYYY-MM-DD or null",
 "legal_areas": ["criminal","civil","family","consumer","cyber","property","labour","constitutional"],
 "search_queries": ["3-5 short queries (in English) to search a database of Indian law sections"]}
Do not give legal conclusions."""


def gen_json(system, contents):
    r = client.models.generate_content(
        model=GEN_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    if not r.text:
        raise RuntimeError("Empty response from model")
    data = json.loads(r.text)
    if isinstance(data, list) and data:
        data = data[0]
    return data


def extract(case):
    try:
        return gen_json(EXTRACT_PROMPT, case)
    except Exception:
        return {"incident_date": None, "search_queries": [case[:300]]}


def retrieve(queries, incident_date, per_query=6, max_total=12):
    where = None
    if incident_date:
        try:
            d = date.fromisoformat(incident_date)
            excluded = OLD_CODES if d >= CUTOFF else NEW_CODES
            where = {"short": {"$nin": excluded}}
        except (ValueError, TypeError):
            pass

    r = client.models.embed_content(
        model=EMBED_MODEL,
        contents=queries,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    res = coll.query(
        query_embeddings=[e.values for e in r.embeddings],
        n_results=per_query,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    best = {}
    for ids, docs, metas, dists in zip(
        res["ids"], res["documents"], res["metadatas"], res["distances"]
    ):
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            if i not in best or dist < best[i]["dist"]:
                best[i] = {"id": i, "doc": doc, "meta": meta, "dist": dist}
    return sorted(best.values(), key=lambda x: x["dist"])[:max_total]


def format_context(chunks):
    return "\n\n".join(
        f'[{c["meta"]["act"]} ({c["meta"]["year"]}) - {c["meta"]["kind"]} '
        f'{c["meta"]["section"]}: {c["meta"]["title"]}] '
        f'(status: {c["meta"]["status"]}; old/new equivalent: '
        f'{c["meta"]["old_equivalent"] or "n/a"})\n{c["doc"]}'
        for c in chunks
    )


def norm_sec(s):
    m = re.search(r"(\d+[A-Za-z]?)", s or "")
    return m.group(1).upper() if m else ""


def act_matches(model_act, meta):
    a = (model_act or "").lower()
    return bool(a) and (
        meta["short"].lower() in a
        or meta["act"].lower() in a
        or a in meta["act"].lower()
    )


def validate(data, chunks):
    """Keep only provisions that exist in the retrieved database records."""
    kept, dropped = [], 0
    for p in data.get("applicable_provisions", []):
        sec = norm_sec(p.get("section_or_article"))
        match = next(
            (
                c
                for c in chunks
                if str(c["meta"]["section"]).upper() == sec
                and act_matches(p.get("act"), c["meta"])
            ),
            None,
        )
        if match:
            p["source_text"] = match["doc"]
            kept.append(p)
        else:
            dropped += 1
    data["applicable_provisions"] = kept
    data["validation"] = {"unverified_citations_removed": dropped}
    return data


def analyze(case, incident_date=None, history=None):
    search_text = case
    if history:
        earlier = " ".join(t["text"] for t in history if t["role"] == "user")[:1500]
        search_text = f"{earlier}\n{case}"
    facts = extract(search_text)
    incident_date = incident_date or facts.get("incident_date")
    queries = facts.get("search_queries") or [search_text[:300]]
    chunks = retrieve(queries + [search_text[:500]], incident_date)

    user_input = case
    if incident_date:
        user_input = f"Incident date: {incident_date}\n{case}"
    if history:
        convo = "\n".join(f'{t["role"]}: {t["text"]}' for t in history)
        user_input = f"Earlier conversation:\n{convo}\n\nNew message:\n{user_input}"

    system = PROMPT.replace("{{retrieved_chunks}}", format_context(chunks)).replace(
        "{{user_input}}", user_input
    )
    data = gen_json(
        system,
        f"<user_case>\n{user_input}\n</user_case>\n\nAnalyse the case above and return the JSON.",
    )
    return validate(data, chunks)