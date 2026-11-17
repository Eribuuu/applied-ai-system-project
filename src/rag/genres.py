"""
Genre names people type vs. the genre names the catalog uses.

The Recommender's genre centroids say which catalog genres *sound* alike, but
they only help once a request names a catalog genre. Users say "rap", "r&b" or
"lofi"; Spotify files those under "hip-hop", "r-n-b" and (closest) "study".
This module bridges the two:

- resolve_genre: map a typed genre (e.g. --genre rap) to a catalog genre.
- genres_in_query: find catalog genres or aliases named in a text query.

Both match on normalized names (lowercase, hyphens as spaces), the same form
Recommender._normalize_genre uses, so results plug straight into scoring.
"""

import re
from typing import Dict, Iterable, List, Set

# catalog genre (normalized) -> other names people use for it
ALIASES: Dict[str, List[str]] = {
    "hip hop": ["rap", "hiphop"],
    "r n b": ["r&b", "rnb", "r and b", "rhythm and blues"],
    "edm": ["electronic dance music"],
    "drum and bass": ["dnb", "d&b", "drum n bass"],
    "rock n roll": ["rock and roll", "rock 'n' roll"],
    "k pop": ["kpop"],
    "j pop": ["jpop"],
    "j rock": ["jrock"],
    "mpb": ["bossa nova"],
    "show tunes": ["broadway", "musicals"],
    "new age": ["meditation"],
    "classical": ["orchestral"],
    # Spotify has no lofi genre; its "study" playlists are mostly lofi beats.
    "study": ["lofi", "lo fi"],
}


def normalize(text: str) -> str:
    """Lowercase, hyphens as spaces, other punctuation (except & and ') dropped."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w&' ]+", " ", text)
    return " ".join(text.split())


def _phrases(known: Set[str]) -> Dict[str, str]:
    """Every phrase that names a known genre -> that genre. A genre's own name wins over an alias."""
    phrases = {alias: genre for genre, names in ALIASES.items() if genre in known for alias in names}
    phrases.update({g: g for g in known})
    return phrases


def resolve_genre(name: str, known: Iterable[str]) -> str:
    """
    The catalog genre a typed name means: itself if the catalog has it, else
    its alias target, else the name unchanged (the Recommender then falls back
    to word overlap).
    """
    return _phrases(set(known)).get(normalize(name), name)


def genres_in_query(query: str, known: Iterable[str]) -> List[str]:
    """
    Catalog genres named in a query, in the order they appear. Longer phrases
    win, so "synth pop" is one match (synth pop), not two (synth pop + pop).
    Whole words only: "studying" doesn't match "study".
    """
    phrases = _phrases(set(known))
    text = f" {normalize(query)} "
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = f" {phrase} "
        pos = text.find(pattern)
        if pos == -1:
            continue
        found.append((pos, phrases[phrase]))
        text = text.replace(pattern, " " + "_" * len(phrase) + " ")  # don't match inside it again
    return list(dict.fromkeys(genre for _, genre in sorted(found)))
