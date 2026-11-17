"""
Hybrid retrieval: dense search over song cards + the Recommender's feature scoring,
fused with Reciprocal Rank Fusion (RRF).

For one request:
  1. Dense search embeds the text query and takes the top ~300 cards by cosine.
  2. A UserProfile is built: targets from CLI flags win; any target left unset
     is filled with the average of the top ~20 dense hits (pseudo-relevance
     feedback), so a text-only query still drives the numeric scorer. If the
     query names exactly one genre ("aggressive rap" -> hip hop, see genres.py),
     that becomes the profile's genre, so the genre centroids apply.
  3. The FeatureScorer scores every song against that profile and takes its top ~300.
  4. RRF merges the two rankings by rank, not raw score (cosine and the 0-1
     weighted score aren't on comparable scales).
  5. Songs whose feature score is below min_score are dropped; the top k are
     returned with their per-retriever evidence and the Recommender's reasons.

With flags and no query, only the feature ranking exists, so results match
Recommender.recommend_with_details.
"""

from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .feature_scorer import FeatureScorer
from .genres import genres_in_query, resolve_genre

try:
    from ..recommender import Recommender, Song, UserProfile
except ImportError:  # run as `python src/main.py`, where src/ is the top level
    from recommender import Recommender, Song, UserProfile

DENSE_N = 300        # candidates taken from each retriever
PRF_N = 20           # dense hits averaged into the pseudo-relevance profile
PRF_GENRE_SHARE = 0.5  # a genre joins the profile only if this share of those hits has it
RRF_K = 60           # standard RRF constant; damps the gap between rank 1 and rank 2
DEFAULT_WEIGHTS = {"dense": 1.0, "feature": 1.0}
DEFAULT_MIN_SCORE = 0.5  # same cut-off as src/main.py


@dataclass
class Hit:
    """One retrieved song and the evidence for it."""
    index: int
    song: Song
    rrf: float
    feature_score: float
    feature_rank: Optional[int]  # 1-based rank in the feature retriever's list, None if absent
    dense_rank: Optional[int]    # 1-based rank in the dense list, None if absent or no query
    dense_sim: Optional[float]
    reasons: List[str]


@dataclass
class RetrievalResult:
    hits: List[Hit]
    profile: UserProfile
    profile_source: str  # e.g. "flags", "pseudo-relevance + query genre", "flags + pseudo-relevance"
    query_genres: List[str]  # catalog genres named in the query (used only when exactly one)


def rrf_fuse(
    rankings: Mapping[str, Sequence[int]],
    weights: Optional[Mapping[str, float]] = None,
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion: each item scores sum(weight / (k + rank)) over the
    rankings it appears in (rank starts at 1). Highest first; ties keep the
    order items were first seen.
    """
    weights = weights or {}
    fused: Dict[int, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, item in enumerate(ranked, 1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def is_empty(profile: Optional[UserProfile]) -> bool:
    return profile is None or all(getattr(profile, f.name) is None for f in fields(UserProfile))


class HybridRetriever:
    def __init__(
        self,
        scorer: FeatureScorer,
        embeddings: Optional[np.ndarray] = None,
        embedder=None,
        dense_n: int = DENSE_N,
        prf_n: int = PRF_N,
        rrf_k: int = RRF_K,
        weights: Optional[Mapping[str, float]] = None,
    ):
        """
        scorer: FeatureScorer over the catalog's songs, in catalog order.
        embeddings: row i = song i's card vector (normalized). With no
            embeddings/embedder, only flag-based (feature) retrieval works.
        """
        if embeddings is not None and len(embeddings) != scorer.n:
            raise ValueError(f"{len(embeddings)} embeddings for {scorer.n} songs")
        self.scorer = scorer
        self.rec: Recommender = scorer.rec
        self.embeddings = embeddings
        self.embedder = embedder
        self.dense_n = dense_n
        self.prf_n = prf_n
        self.rrf_k = rrf_k
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.known_genres = set(scorer.genre_vocab)

    @classmethod
    def from_disk(cls, **kwargs) -> "HybridRetriever":
        """The full Spotify catalog with its saved index (see src/rag/index.py)."""
        from .data import load_catalog, songs_from_frame
        from .index import BgeEmbedder, load_index

        catalog = load_catalog()
        scorer = FeatureScorer(Recommender(songs_from_frame(catalog)), catalog["genres"].tolist())
        return cls(scorer, load_index(catalog), BgeEmbedder(), **kwargs)

    # --- the two retrievers ---------------------------------------------------

    def dense(self, query: str, n: int, mask: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """Top n (song index, cosine similarity) for a text query."""
        if self.embeddings is None or self.embedder is None:
            raise RuntimeError("text queries need the embedding index; build it with python -m src.rag.index")
        sims = self.embeddings @ self.embedder.embed_query(query)
        return FeatureScorer.rank(sims, n, min_score=-np.inf, mask=mask)

    def profile_from_hits(self, indices: Sequence[int]) -> UserProfile:
        """Average audio features of the given songs; their genre too if most share it."""
        idx = np.asarray(indices, dtype=np.int64)
        targets = {
            user_attr: float(self.scorer.features[name][idx].mean())
            for name, (user_attr, _) in Recommender.NUMERIC_FEATURES.items()
        }
        genre, count = Counter(self.rec.songs[i].genre for i in idx).most_common(1)[0]
        if count / len(idx) < PRF_GENRE_SHARE:
            genre = None
        return UserProfile(favorite_genre=genre, **targets)

    def query_genre(self, query: str) -> Tuple[Optional[str], List[str]]:
        """(the genre to use, every genre named). Used only when the query names exactly one."""
        named = genres_in_query(query, self.known_genres) if query else []
        return (named[0] if len(named) == 1 else None), named

    def build_profile(
        self,
        flags: Optional[UserProfile],
        dense_hits: Sequence[Tuple[int, float]],
        query_genre: Optional[str] = None,
    ) -> Tuple[UserProfile, str]:
        """
        Priority: flags > the genre named in the query > the average of the top
        dense hits. Without dense hits (no query) the flags are the whole profile.
        """
        if not dense_hits:
            return flags, "flags"
        prf = self.profile_from_hits([i for i, _ in dense_hits[: self.prf_n]])
        source = "pseudo-relevance"
        if query_genre is not None:
            prf = replace(prf, favorite_genre=query_genre)
            source += " + query genre"
        if is_empty(flags):
            return prf, source
        given = {f.name: getattr(flags, f.name) for f in fields(UserProfile) if getattr(flags, f.name) is not None}
        return replace(prf, **given), "flags + " + source

    # --- the full pipeline ----------------------------------------------------

    def retrieve(
        self,
        query: Optional[str] = None,
        flags: Optional[UserProfile] = None,
        k: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
        mask: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        """Top k songs for a text query, explicit feature targets, or both."""
        query = (query or "").strip()
        if not query and is_empty(flags):
            raise ValueError("give a text query, at least one feature target, or both")
        if flags is not None and flags.favorite_genre:
            flags = replace(flags, favorite_genre=resolve_genre(flags.favorite_genre, self.known_genres))

        genre, named = self.query_genre(query)
        dense_hits = self.dense(query, self.dense_n, mask) if query else []
        profile, source = self.build_profile(flags, dense_hits, genre)

        closeness = self.scorer.closeness(profile)
        feature_scores = self.scorer.weighted_scores(closeness)
        feature_hits = FeatureScorer.rank(feature_scores, self.dense_n, min_score, mask)

        rankings = {"feature": [i for i, _ in feature_hits]}
        if dense_hits:
            rankings["dense"] = [i for i, _ in dense_hits]
        fused = rrf_fuse(rankings, self.weights, self.rrf_k)

        dense_pos = {i: (r, sim) for r, (i, sim) in enumerate(dense_hits, 1)}
        feature_pos = {i: r for r, (i, _) in enumerate(feature_hits, 1)}

        hits: List[Hit] = []
        for i, rrf in fused:
            if feature_scores[i] < min_score:
                continue
            song = self.rec.songs[i]
            d_rank, d_sim = dense_pos.get(i, (None, None))
            hits.append(Hit(
                index=i,
                song=song,
                rrf=rrf,
                feature_score=float(feature_scores[i]),
                feature_rank=feature_pos.get(i),
                dense_rank=d_rank,
                dense_sim=d_sim,
                reasons=self.rec.reasons(profile, song, self.scorer.song_closeness(closeness, i)),
            ))
            if len(hits) == k:
                break
        return RetrievalResult(hits=hits, profile=profile, profile_source=source, query_genres=named)
