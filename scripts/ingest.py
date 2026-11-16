"""Reads data/laws.jsonl, embeds each record with Gemini, stores in ChromaDB.

Run from the project root:  python scripts/ingest.py
Re-run whenever you change data/laws.jsonl.
"""
import json
import time
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

import chromadb
from google import genai
from google.genai import types

client = genai.Client()  # reads GEMINI_API_KEY from the environment
EMBED_MODEL = "gemini-embedding-001"
BATCH = 50


def embed(texts):
    for attempt in range(5):
        try:
            r = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in r.embeddings]
        except Exception as e:
            print("Embedding failed, retrying:", e)
            time.sleep(2 ** attempt)
    raise RuntimeError("Embedding failed after 5 attempts")


def main():
    path = BASE / "data" / "laws.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(records)} records from {path}")

    db = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
    try:
        db.delete_collection("indian_laws")
    except Exception:
        pass
    coll = db.create_collection("indian_laws", metadata={"hnsw:space": "cosine"})

    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        to_embed = [
            f'{r["act"]} ({r["year"]}) {r["kind"]} {r["section"]}: {r["title"]}\n{r["text"]}'[:6000]
            for r in batch
        ]
        coll.add(
            ids=[f'{r["short"]}-{r["section"]}' for r in batch],
            embeddings=embed(to_embed),
            documents=[r["text"] for r in batch],
            metadatas=[
                {
                    k: r.get(k, "")
                    for k in (
                        "act", "short", "year", "kind",
                        "section", "title", "status", "old_equivalent",
                    )
                }
                for r in batch
            ],
        )
        print(f"Ingested {i + len(batch)}/{len(records)}")
        time.sleep(1)
    print("Done.")


if __name__ == "__main__":
    main()