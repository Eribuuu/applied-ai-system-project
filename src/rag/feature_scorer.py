"""
Vectorized version of the Recommender's scoring, for catalogs too big to loop over.

Recommender.recommend_with_details scores one song at a time in Python, which
is fine for 20 songs but slow for ~81k. FeatureScorer does the same math on
NumPy arrays, one feature at a time across every song at once. It reuses the
Recommender's WEIGHTS, SIGMAS, NUMERIC_FEATURES and genre similarity, so the
scores match the Recommender's exactly when each song has one genre.

The one difference: a song filed under several genres gets the best genre
match among them, not just the first one's.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ..recommender import Recommender, UserProfile
except ImportError:  # run as `python src/main.py`, where src/ is the top level
    from recommender import Recommender, UserProfile


class FeatureScorer:
    def __init__(self, recommender: Recommender, genres: Optional[Sequence[Iterable[str]]] = None):
        """
        recommender: built on the catalog's songs, in catalog order.
        genres: every genre each song is filed under, aligned with
            recommender.songs. Defaults to each song's single `genre`.
        """
        self.rec = recommender
        songs = recommender.songs
        self.n = len(songs)

        self.features: Dict[str, np.ndarray] = {
            name: np.array([getattr(s, song_attr) for s in songs], dtype=float)
            for name, (_, song_attr) in Recommender.NUMERIC_FEATURES.items()
        }

        if genres is None:
            genres = [[s.genre] for s in songs]
        if len(genres) != self.n:
            raise ValueError(f"got genres for {len(genres)} songs, expected {self.n}")

        # Flatten song -> genres into parallel (song index, genre code) arrays, so a
        # genre target is scored once per distinct genre, then spread to songs.
        vocab: Dict[str, int] = {}
        song_idx, genre_code = [], []
        for i, gs in enumerate(genres):
            for g in gs:
                code = vocab.setdefault(Recommender._normalize_genre(g), len(vocab))
                song_idx.append(i)
                genre_code.append(code)
        self.genre_vocab: List[str] = list(vocab)
        self._song_idx = np.array(song_idx, dtype=np.int64)
        self._genre_code = np.array(genre_code, dtype=np.int64)

    def genre_closeness(self, target: str) -> np.ndarray:
        """Best genre match per song, 0 to 1."""
        per_genre = np.array([self.rec.genre_closeness(g, target) for g in self.genre_vocab])
        best = np.zeros(self.n)
        np.maximum.at(best, self._song_idx, per_genre[self._genre_code])
        return best

    def closeness(self, user: UserProfile) -> Dict[str, np.ndarray]:
        """
        Per-feature closeness arrays (one value per song) for every preference
        the user set, in the same order as Recommender.closeness.
        """
        result: Dict[str, np.ndarray] = {}
        if user.favorite_genre is not None:
            result["genre"] = self.genre_closeness(user.favorite_genre)
        for name, (user_attr, _) in Recommender.NUMERIC_FEATURES.items():
            target = getattr(user, user_attr)
            if target is not None:
                sigma = Recommender.SIGMAS[name]
                result[name] = np.exp(-((self.features[name] - target) ** 2) / (2 * sigma ** 2))
        return result

    def weighted_scores(self, closeness: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average per song, as in Recommender.weighted_score. No preferences = all 0."""
        total_weight = sum(Recommender.WEIGHTS[name] for name in closeness)
        if total_weight == 0:
            return np.zeros(self.n)
        total = np.zeros(self.n)
        for name, c in closeness.items():
            total += Recommender.WEIGHTS[name] * c
        return total / total_weight

    def scores(self, user: UserProfile) -> np.ndarray:
        """Score (0 to 1) for every song in catalog order."""
        return self.weighted_scores(self.closeness(user))

    def top(
        self,
        user: UserProfile,
        n: int,
        min_score: float = 0.0,
        mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """
        The n best (song index, score) pairs, highest first. Ties keep catalog
        order, like the Recommender's stable sort. `mask` (bool per song)
        limits which songs are eligible.
        """
        s = self.scores(user)
        eligible = s >= min_score
        if mask is not None:
            eligible &= mask
        idx = np.flatnonzero(eligible)
        order = idx[np.argsort(-s[idx], kind="stable")][:n]
        return [(int(i), float(s[i])) for i in order]

    def song_closeness(self, closeness: Dict[str, np.ndarray], i: int) -> Dict[str, float]:
        """One song's closeness dict, the shape Recommender.reasons() expects."""
        return {name: float(c[i]) for name, c in closeness.items()}
