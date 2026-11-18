"""
Command line runner for the Music Recommender Simulation.

With no arguments it starts interactive mode: numbered menus for describing
what you want and picking genre/sound settings (see src/interactive.py).

--demo runs the original demo: the three PROFILES below against the 20-song
data/songs.csv.

With --query and/or feature flags it searches the full Spotify catalog with
hybrid retrieval (src/rag) and has a local LLM explain the picks, for scripting:

    python src/main.py --query "mellow songs for late-night studying"
    python src/main.py --query "rap for the gym" --energy 0.9
    python src/main.py --genre rock --energy 0.9 --valence 0.3      # flags only: no LLM
    python src/main.py --list-genres

Text queries need the index (python -m src.rag.data && python -m src.rag.index).
Explanations need Ollama running (ollama serve) with qwen2.5:3b pulled;
without it, songs are explained with the template reasons instead.

The recommendation logic lives in recommender.py:
- load_songs
- Recommender (score, recommend, explain_recommendation)
"""

import argparse
import difflib
import sys
from dataclasses import asdict
from typing import List, Optional, TextIO

try:
    from .recommender import Recommender, UserProfile, load_songs
except ImportError:  # run as `python src/main.py`
    from recommender import Recommender, UserProfile, load_songs


# Taste profiles: the target values each song is compared against.
# Set a target to None to leave that feature out of scoring.
#   energy:       0.0 (calm) – 1.0 (intense)
#   valence:      0.0 (dark/sad) – 1.0 (bright/happy)
#   danceability: 0.0 – 1.0
#   tempo_bpm:    beats per minute
#   acousticness: 0.0 (electronic) – 1.0 (acoustic)
PROFILES = {
    "High-Energy Pop": UserProfile(
        favorite_genre="pop",
        target_energy=0.92,
        target_valence=0.75,
        target_danceability=0.88,
        target_tempo_bpm=130,
        target_acousticness=0.05,
    ),
    "Chill Lofi": UserProfile(
        favorite_genre="lofi",
        target_energy=0.38,
        target_valence=0.58,
        target_danceability=0.60,
        target_tempo_bpm=76,
        target_acousticness=0.80,
    ),
    "Deep Intense Rock": UserProfile(
        favorite_genre="rock",
        target_energy=0.92,
        target_valence=0.40,
        target_danceability=0.60,
        target_tempo_bpm=150,
        target_acousticness=0.08,
    ),
}


# Songs scoring below this are treated as "not a good match" and are not shown.
MIN_SCORE = 0.5


def print_recommendations(recommender: Recommender, name: str, user: UserProfile, k: int = 5) -> None:
    print("=" * 60)
    print(f"Profile: {name}")
    print("=" * 60)
    for field, value in asdict(user).items():
        print(f"  {field}: {value}")

    results = recommender.recommend_with_details(user, k=k, min_score=MIN_SCORE)

    print("\nTop recommendations:\n")
    if not results:
        print(f"No songs scored at least {MIN_SCORE:.2f}; nothing in the catalog is a good match.\n")
        return
    for rec in results:
        print(f"{rec.song.title} - Score: {rec.score:.2f}")
        print(f"Because: {', '.join(rec.reasons)}")
        print()
    if len(results) < k:
        print(f"(Only {len(results)} of {k} songs scored at least {MIN_SCORE:.2f}.)\n")


def run_demo() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded Songs: {len(songs)}\n")

    recommender = Recommender(songs)

    for name, user in PROFILES.items():
        print_recommendations(recommender, name, user)


# --- RAG query mode ---------------------------------------------------------------

def unit_interval(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
    return value


def positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be above 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Music recommender. No arguments: interactive menus. "
                    "With --query and/or feature flags: one search over the Spotify catalog.",
    )
    p.add_argument("--demo", action="store_true", help="run the original 20-song demo and exit")
    p.add_argument("--query", "-q", help='what you want to hear, e.g. "mellow songs for late-night studying"')
    p.add_argument("-k", type=int, default=5, help="number of songs (default 5)")
    target = p.add_argument_group("feature targets (each optional; set ones override what the query implies)")
    target.add_argument("--genre", help="genre, e.g. rock, hip-hop, rap, lofi (see --list-genres)")
    target.add_argument("--energy", type=unit_interval, help="0 calm .. 1 intense")
    target.add_argument("--valence", type=unit_interval, help="0 dark/sad .. 1 bright/happy")
    target.add_argument("--danceability", type=unit_interval, help="0 .. 1")
    target.add_argument("--tempo", type=positive, help="beats per minute")
    target.add_argument("--acousticness", type=unit_interval, help="0 electronic .. 1 acoustic")
    p.add_argument("--min-score", type=unit_interval, default=MIN_SCORE,
                   help=f"drop songs whose feature score is below this (default {MIN_SCORE})")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM; explain with template reasons")
    p.add_argument("--model", default=None, help="Ollama model for explanations (default qwen2.5:3b)")
    p.add_argument("--verbose", "-v", action="store_true", help="show the profile used and guardrail notes")
    p.add_argument("--list-genres", action="store_true", help="print the catalog's genres and exit")
    return p


def flags_from_args(args: argparse.Namespace) -> Optional[UserProfile]:
    """The feature targets the user set, or None if they set none."""
    profile = UserProfile(
        favorite_genre=args.genre,
        target_energy=args.energy,
        target_valence=args.valence,
        target_danceability=args.danceability,
        target_tempo_bpm=args.tempo,
        target_acousticness=args.acousticness,
    )
    return None if all(v is None for v in asdict(profile).values()) else profile


def genre_warning(typed: str, known: List[str]) -> Optional[str]:
    """A note when a typed genre (after aliases) isn't in the catalog, with close suggestions."""
    try:
        from .rag.genres import normalize, resolve_genre
    except ImportError:
        from rag.genres import normalize, resolve_genre
    if normalize(resolve_genre(typed, known)) in known:
        return None
    close = difflib.get_close_matches(normalize(typed), known, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else " See --list-genres."
    return f"Note: '{typed}' isn't a genre in this catalog, so genre only matches by shared words.{hint}"


def describe_profile(profile: UserProfile) -> str:
    parts = []
    if profile.favorite_genre:
        parts.append(f"genre {profile.favorite_genre}")
    for label, value, fmt in [
        ("energy", profile.target_energy, "{:.2f}"),
        ("valence", profile.target_valence, "{:.2f}"),
        ("danceability", profile.target_danceability, "{:.2f}"),
        ("tempo", profile.target_tempo_bpm, "{:.0f} BPM"),
        ("acousticness", profile.target_acousticness, "{:.2f}"),
    ]:
        if value is not None:
            parts.append(f"{label} {fmt.format(value)}")
    return ", ".join(parts)


def print_results(
    query: str, result, explanation, verbose: bool, llm_requested: bool = True,
    genres: Optional[List[List[str]]] = None, out: TextIO = sys.stdout,
) -> None:
    """genres: every genre per catalog song (by index); defaults to each song's main genre."""
    w = lambda line="": print(line, file=out)  # noqa: E731
    w("=" * 60)
    w(f'Query: "{query}"' if query else "Feature targets only")
    if verbose:
        w(f"Profile ({result.profile_source}): {describe_profile(result.profile)}")
    w("=" * 60)
    if not result.hits:
        w("No songs passed the match threshold. Try a broader query or a lower --min-score.")
        return
    if explanation.summary:
        w(explanation.summary)
    w()
    for n, item in enumerate(explanation.songs, 1):
        song, hit = item.hit.song, item.hit
        song_genres = ", ".join(genres[hit.index]) if genres else song.genre
        w(f"{n}. {song.title} — {song.artist}  [{song_genres}]  match {hit.feature_score:.2f}")
        w(f"   {item.text}")
    templated = sum(s.source == "template" for s in explanation.songs)
    if query and llm_requested and templated:
        if any(i.startswith("LLM unavailable") for i in explanation.issues):
            reason = "the local LLM isn't reachable; start it with `ollama serve`"
        elif not explanation.used_llm:
            reason = "the LLM reply couldn't be used"
        else:
            reason = "the rest failed the guardrail"
        w(f"\n({templated} of {len(explanation.songs)} explained with template reasons: {reason}. "
          f"Use -v for details.)")
    if verbose and explanation.issues:
        w("\nExplanation notes:")
        for issue in explanation.issues:
            w(f"  - {issue}")


def run_query(args: argparse.Namespace) -> int:
    try:
        from .rag.data import load_catalog
        from .rag.generate import Explainer, OllamaClient
        from .rag.retrieve import HybridRetriever
    except ImportError:
        from rag.data import load_catalog
        from rag.generate import Explainer, OllamaClient
        from rag.retrieve import HybridRetriever

    query = (args.query or "").strip()
    flags = flags_from_args(args)

    print("Loading catalog and index...", file=sys.stderr)
    try:
        retriever = HybridRetriever.from_disk()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.genre:
        note = genre_warning(args.genre, sorted(retriever.known_genres))
        if note:
            print(note, file=sys.stderr)

    result = retriever.retrieve(query=query, flags=flags, k=args.k, min_score=args.min_score)

    rows = load_catalog().to_dict("records")
    if args.no_llm:
        query_for_llm = ""  # template reasons only
    else:
        query_for_llm = query
        if query and result.hits:
            print("Writing explanations with the local LLM...", file=sys.stderr)
    client = OllamaClient(model=args.model) if args.model else None
    explanation = Explainer(rows, client).explain(query_for_llm, result)
    print_results(query, result, explanation, args.verbose, llm_requested=not args.no_llm,
                  genres=[r["genres"] for r in rows])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_genres:
        try:
            from .rag.data import load_catalog
        except ImportError:
            from rag.data import load_catalog
        genres = sorted({g for gs in load_catalog()["genres"] for g in gs})
        print(f"{len(genres)} genres:\n" + ", ".join(genres))
        print("Also accepted: rap, r&b, lofi, kpop, dnb, bossa nova, and other common aliases.")
        return 0
    if args.demo:
        run_demo()
        return 0
    if args.query is None and flags_from_args(args) is None:
        try:
            from .interactive import run_interactive
        except ImportError:
            from interactive import run_interactive
        return run_interactive()
    if args.query is not None and not args.query.strip() and flags_from_args(args) is None:
        build_parser().error("--query is empty; give some words or a feature flag")
    return run_query(args)


if __name__ == "__main__":
    sys.exit(main())
