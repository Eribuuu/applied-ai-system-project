"""
Embedding index: one normalized vector per song card, saved next to the catalog.

Build (or rebuild) it from the project root with:
    python -m src.rag.index           # skips if the saved index is current
    python -m src.rag.index --force   # always rebuild

meta.json records the model and a hash of every card, so a changed catalog,
changed descriptor bins, or a different model is detected and the stale index
is never used with the wrong data.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterator, List, Protocol, Sequence

import numpy as np
import pandas as pd

from .cards import make_card
from .data import INDEX_DIR, load_catalog

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge-v1.5's instruction for short retrieval queries; cards (documents) get no prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"
MODEL_CACHE = INDEX_DIR / "models"


class Embedder(Protocol):
    """What the index needs from an embedding model (a fake one works in tests)."""

    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a plain dot product."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class BgeEmbedder:
    """bge-small-en-v1.5 through fastembed (ONNX, CPU, no PyTorch)."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = MODEL_CACHE):
        from fastembed import TextEmbedding  # imported here so tests don't need the model

        self.name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        vectors = list(_progress(self.model.embed(texts, batch_size=batch_size), total=len(texts)))
        return normalize(np.stack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        (vector,) = self.model.embed([QUERY_PREFIX + text])
        return normalize(np.asarray(vector))


def _progress(vectors: Iterator[np.ndarray], total: int, every: int = 10_000) -> Iterator[np.ndarray]:
    """Pass vectors through, printing a line every `every` of them on large jobs."""
    start = time.time()
    for i, v in enumerate(vectors, 1):
        if total >= every and (i % every == 0 or i == total):
            print(f"  embedded {i:,}/{total:,}  ({time.time() - start:.0f}s)", flush=True)
        yield v


def make_cards(catalog: pd.DataFrame) -> List[str]:
    return [make_card(row) for row in catalog.to_dict("records")]


def cards_hash(cards: Sequence[str]) -> str:
    h = hashlib.sha256()
    for card in cards:
        h.update(card.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_index(catalog: pd.DataFrame, embedder: Embedder, out_dir: Path = INDEX_DIR) -> np.ndarray:
    """Embed every card, save embeddings.npy + meta.json, and return the matrix."""
    cards = make_cards(catalog)
    embeddings = embedder.embed_documents(cards)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / EMBEDDINGS_FILE, embeddings)
    meta = {
        "model": embedder.name,
        "n_songs": len(cards),
        "dim": int(embeddings.shape[1]),
        "cards_sha256": cards_hash(cards),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return embeddings


def index_is_current(catalog: pd.DataFrame, model_name: str, out_dir: Path = INDEX_DIR) -> bool:
    meta_path, emb_path = out_dir / META_FILE, out_dir / EMBEDDINGS_FILE
    if not meta_path.exists() or not emb_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return (
        meta.get("model") == model_name
        and meta.get("n_songs") == len(catalog)
        and meta.get("cards_sha256") == cards_hash(make_cards(catalog))
    )


def load_index(catalog: pd.DataFrame, model_name: str = MODEL_NAME, out_dir: Path = INDEX_DIR) -> np.ndarray:
    """The saved embeddings, or an error telling you to rebuild when they're stale."""
    if not index_is_current(catalog, model_name, out_dir):
        raise FileNotFoundError(
            f"No current embedding index in {out_dir}. Build it with: python -m src.rag.index"
        )
    return np.load(out_dir / EMBEDDINGS_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the song-card embedding index.")
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    args = parser.parse_args()

    catalog = load_catalog()
    if not args.force and index_is_current(catalog, MODEL_NAME):
        print(f"Index in {INDEX_DIR} is current ({len(catalog):,} songs). Use --force to rebuild.")
        return

    print(f"Embedding {len(catalog):,} song cards with {MODEL_NAME} ...")
    start = time.time()
    embeddings = build_index(catalog, BgeEmbedder())
    print(f"Saved {INDEX_DIR / EMBEDDINGS_FILE} {embeddings.shape} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
