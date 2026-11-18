"""
Compare small local LLMs (via Ollama) on the generation step.

Every model gets the same prompt and the same retrieved songs for each query,
so differences come from the model alone. Automatic checks per answer:
  json      the reply parses and has the expected fields
  ids       every cited id is one of the retrieved songs, none repeated
  coverage  share of retrieved songs the reply explains
  invented  numbers in the reply that appear nowhere in the context (a hallucination signal)
  seconds   generation time, excluding model load
The full answers are written to scripts/results/llm_comparison.md for reading.

Needs a running Ollama server (ollama serve) with the models pulled. Run from the project root:
    python scripts/compare_llms.py qwen2.5:3b llama3.2:3b
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.cards import descriptors  # noqa: E402
from src.rag.data import load_catalog  # noqa: E402
from src.rag.retrieve import HybridRetriever  # noqa: E402

OLLAMA = "http://localhost:11434"
LABELS = Path("tests/data/labeled_queries.json")
OUT = Path("scripts/results/llm_comparison.md")
QUERIES = 5
K = 5

SYSTEM = (
    "You explain music recommendations. You get a listener's request and a numbered list of "
    "songs a recommender already picked, with facts about each.\n"
    "Rules:\n"
    "- Use only the facts given. Do not invent numbers, years, artists, lyrics or other details.\n"
    "- Only discuss the listed songs, referring to each by its id (S1, S2, ...).\n"
    "- For every song, write one or two sentences on why it fits the request.\n"
    "- Start with one short sentence summarizing the picks.\n"
    "Reply with JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "why": {"type": "string"}},
                "required": ["id", "why"],
            },
        },
    },
    "required": ["summary", "picks"],
}


def build_context(query: str, hits, rows: List[Dict]) -> str:
    lines = [f'Listener request: "{query}"', "", "Retrieved songs:"]
    for n, h in enumerate(hits, 1):
        r = rows[h.index]
        lines.append(
            f'[S{n}] "{r["title"]}" by {r["artist"]}. Genres: {", ".join(r["genres"])}. '
            f'Sound: {", ".join(descriptors(r))}. '
            f'Match details: {", ".join(h.reasons)}. Match score: {h.feature_score:.2f}.'
        )
    return "\n".join(lines)


def chat(model: str, user: str) -> Dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
    }
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check(reply_text: str, context: str, k: int) -> Dict:
    result = {"json": False, "ids": False, "coverage": 0.0, "invented": []}
    try:
        reply = json.loads(reply_text)
        picks = reply["picks"]
        ids = [p["id"].strip() for p in picks]
        texts = [reply["summary"]] + [p["why"] for p in picks]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return result
    result["json"] = True
    valid = {f"S{i}" for i in range(1, k + 1)}
    result["ids"] = set(ids) <= valid and len(ids) == len(set(ids))
    result["coverage"] = len(set(ids) & valid) / k
    known = set(NUMBER.findall(context)) | {str(i) for i in range(1, k + 1)}
    result["invented"] = sorted({n for t in texts for n in NUMBER.findall(t)} - known)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local LLMs on the explanation step.")
    parser.add_argument("models", nargs="+", help="Ollama model tags, e.g. qwen2.5:3b")
    args = parser.parse_args()

    rows = load_catalog().to_dict("records")
    retriever = HybridRetriever.from_disk()
    queries = [q["query"] for q in json.loads(LABELS.read_text())[:QUERIES]]
    contexts = [build_context(q, retriever.retrieve(query=q, k=K).hits, rows) for q in queries]

    report = ["# Local LLM comparison", "", f"{len(queries)} queries, top {K} songs each, temperature 0.", ""]
    summary = []
    for model in args.models:
        load = chat(model, "Reply with {}")  # warm-up: loads the model so timings exclude it
        print(f"\n== {model}  (load {load.get('load_duration', 0) / 1e9:.1f}s)")
        report += [f"## {model}", ""]
        stats = {"json": 0, "ids": 0, "coverage": [], "invented": 0, "seconds": [], "tok_s": []}
        for q, ctx in zip(queries, contexts):
            start = time.time()
            resp = chat(model, ctx)
            seconds = time.time() - start
            text = resp["message"]["content"]
            c = check(text, ctx, K)
            tok_s = resp.get("eval_count", 0) / max(resp.get("eval_duration", 1) / 1e9, 1e-9)
            stats["json"] += c["json"]
            stats["ids"] += c["ids"]
            stats["coverage"].append(c["coverage"])
            stats["invented"] += bool(c["invented"])
            stats["seconds"].append(seconds)
            stats["tok_s"].append(tok_s)
            print(f"  {q[:42]:<42} json={c['json']!s:<5} ids={c['ids']!s:<5} "
                  f"cover={c['coverage']:.0%} invented={c['invented'] or '-'}  {seconds:.1f}s")
            report += [f"### {q}", "", f"json={c['json']}, ids={c['ids']}, coverage={c['coverage']:.0%}, "
                       f"invented={c['invented'] or 'none'}, {seconds:.1f}s, {tok_s:.1f} tok/s", "",
                       "```json", text.strip(), "```", ""]
        n = len(queries)
        summary.append((model, stats["json"], stats["ids"], sum(stats["coverage"]) / n,
                        stats["invented"], sum(stats["seconds"]) / n, sum(stats["tok_s"]) / n, n))

    table = ["| model | valid JSON | valid ids | coverage | answers w/ invented numbers | avg s | tok/s |",
             "|---|---|---|---|---|---|---|"]
    for m, j, i, cov, inv, sec, tps, n in summary:
        table.append(f"| {m} | {j}/{n} | {i}/{n} | {cov:.0%} | {inv}/{n} | {sec:.1f} | {tps:.1f} |")
    print("\n" + "\n".join(table))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(report[:4] + table + [""] + report[4:]))
    print(f"\nFull answers: {OUT}")


if __name__ == "__main__":
    main()
