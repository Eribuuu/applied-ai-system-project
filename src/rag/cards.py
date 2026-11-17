"""
Song cards: a short text description of each song, for the embedding index.

Embedding models understand words, not numeric closeness ("energy 0.35" and
"energy 0.36" don't embed as near each other), so each audio feature is binned
into descriptor words. A card never contains raw feature numbers; the numeric
fit is the FeatureScorer's job.

Example:
    "Comedy by Gen Hoshino. Genres: acoustic, j pop, singer songwriter.
     Sound: moderate energy, upbeat, positive, danceable, mid-tempo."
"""

from typing import Any, List, Mapping, Optional, Sequence, Tuple

# (upper bound, words): the first bin whose bound the value is below wins.
ENERGY_BINS: Sequence[Tuple[float, str]] = [
    (0.2, "very calm, quiet"),
    (0.4, "calm, mellow"),
    (0.6, "moderate energy"),
    (0.8, "energetic"),
    (float("inf"), "high-energy, intense"),
]
VALENCE_BINS: Sequence[Tuple[float, str]] = [
    (0.2, "dark, sad"),
    (0.4, "melancholic, moody"),
    (0.6, "bittersweet"),
    (0.8, "upbeat, positive"),
    (float("inf"), "happy, cheerful"),
]
TEMPO_BINS: Sequence[Tuple[float, str]] = [
    (80, "slow tempo"),
    (110, "mid-tempo"),
    (140, "upbeat tempo"),
    (float("inf"), "fast tempo"),
]


def _bin(value: float, bins: Sequence[Tuple[float, str]]) -> str:
    for upper, words in bins:
        if value < upper:
            return words
    return bins[-1][1]


def _present(row: Mapping[str, Any], key: str) -> Optional[float]:
    """A column's value, or None when the catalog has no such column (e.g. songs.csv)."""
    value = row.get(key)
    if value is None or value != value:  # missing or NaN
        return None
    return value


def descriptors(row: Mapping[str, Any]) -> List[str]:
    """Descriptor phrases for one catalog row, energy and mood first."""
    words = [
        _bin(row["energy"], ENERGY_BINS),
        _bin(row["valence"], VALENCE_BINS),
    ]

    dance = row["danceability"]
    if dance >= 0.75:
        words.append("very danceable, groovy")
    elif dance >= 0.6:
        words.append("danceable")
    elif dance < 0.35:
        words.append("not for dancing")

    acoustic = row["acousticness"]
    if acoustic >= 0.75:
        words.append("acoustic")
    elif acoustic < 0.15:
        words.append("electric, studio-produced")

    words.append(_bin(row["tempo_bpm"], TEMPO_BINS))

    instrumental = _present(row, "instrumentalness")
    if instrumental is not None and instrumental >= 0.5:
        words.append("instrumental, no vocals")

    speech = _present(row, "speechiness")
    if speech is not None and speech >= 0.66:
        words.append("spoken word")
    elif speech is not None and speech >= 0.33:
        words.append("rap or spoken vocals")

    if _present(row, "explicit"):
        words.append("explicit lyrics")
    return words


def make_card(row: Mapping[str, Any]) -> str:
    """The text that gets embedded for one song."""
    genres = ", ".join(g.replace("-", " ") for g in row["genres"])
    return (
        f"{row['title']} by {row['artist']}. "
        f"Genres: {genres}. "
        f"Sound: {', '.join(descriptors(row))}."
    )
