"""
Interactive, menu-driven front end for the recommender (numbered menus, no extra dependencies).

Started by `python src/main.py` with no arguments. Sound settings are chosen as
words ("calm", "energetic") rather than numbers; LEVELS maps them to the same
targets the --energy/--valence/... flags take. The catalog and index load once,
on the first search, and are reused for every search after that.
"""

import difflib
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, TextIO, Tuple

try:
    from .main import describe_profile, print_results, run_demo
    from .recommender import UserProfile
except ImportError:  # run as `python src/main.py`
    from main import describe_profile, print_results, run_demo
    from recommender import UserProfile

# Menu choices for each sound setting: (label, UserProfile attribute, [(word, target value), ...]).
# Targets sit in the middle of the matching card descriptor bins (src/rag/cards.py).
LEVELS: List[Tuple[str, str, List[Tuple[str, float]]]] = [
    ("Energy", "target_energy",
     [("very calm", 0.1), ("calm", 0.3), ("moderate", 0.5), ("energetic", 0.7), ("very intense", 0.9)]),
    ("Mood", "target_valence",
     [("dark / sad", 0.1), ("melancholic", 0.3), ("bittersweet", 0.5), ("upbeat", 0.7), ("happy", 0.9)]),
    ("Danceability", "target_danceability",
     [("not for dancing", 0.25), ("some groove", 0.5), ("danceable", 0.67), ("very danceable", 0.85)]),
    ("Sound", "target_acousticness",
     [("electric / produced", 0.05), ("mixed", 0.45), ("acoustic", 0.85)]),
    ("Tempo", "target_tempo_bpm",
     [("slow (~70 BPM)", 70), ("mid-tempo (~95 BPM)", 95), ("upbeat (~125 BPM)", 125), ("fast (~155 BPM)", 155)]),
]

EMPTY = UserProfile(None, None, None, None, None, None)


class Quit(Exception):
    """The user asked to leave (0 at the main menu, Ctrl+D, or Ctrl+C)."""


class Prompter:
    """Numbered-menu input/output; swap input_fn and out to script it in tests."""

    def __init__(self, input_fn: Callable[[str], str] = input, out: TextIO = sys.stdout):
        self.input_fn, self.out = input_fn, out

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def ask(self, prompt: str) -> str:
        try:
            return self.input_fn(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            raise Quit()

    def choose(self, title: str, options: Sequence[str], zero: Optional[str] = None) -> int:
        """
        Show a numbered menu and return the chosen option's index (0-based).
        `zero` labels an extra choice "0" (skip, back, quit), returned as -1.
        """
        self.say(f"\n{title}")
        for i, label in enumerate(options, 1):
            self.say(f"  {i}) {label}")
        if zero:
            self.say(f"  0) {zero}")
        while True:
            answer = self.ask("> ")
            if answer == "0" and zero:
                return -1
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.say(f"Please enter a number from {0 if zero else 1} to {len(options)}.")


@dataclass
class Preferences:
    k: int = 5
    use_llm: bool = True
    min_score: float = 0.5


@dataclass
class LastSearch:
    query: str = ""
    flags: UserProfile = field(default_factory=lambda: EMPTY)
    result: object = None
    explanation: object = None


class InteractiveApp:
    def __init__(
        self,
        prompter: Prompter,
        load_engine: Optional[Callable[[], Tuple[object, list]]] = None,
        llm_client=None,
    ):
        """
        load_engine: returns (HybridRetriever, catalog rows); defaults to the full
            Spotify catalog from disk. Called once, on the first search.
        llm_client: the explanation LLM; defaults to qwen2.5:3b via Ollama.
        """
        self.p = prompter
        self._load_engine = load_engine or default_engine
        self.llm_client = llm_client
        self.engine: Optional[Tuple[object, list]] = None
        self.prefs = Preferences()
        self.last = LastSearch()

    # --- engine ---------------------------------------------------------------

    def get_engine(self):
        if self.engine is None:
            self.p.say("Loading catalog and index (once per session)...")
            self.engine = self._load_engine()
        return self.engine

    def catalog_genres(self) -> List[str]:
        _, rows = self.get_engine()
        return sorted({g for r in rows for g in r["genres"]})

    # --- main loop --------------------------------------------------------------

    def run(self) -> None:
        self.p.say("Music Recommender: interactive mode")
        self.p.say("Type the number of an option and press Enter. Ctrl+C quits at any time.")
        try:
            while True:
                choice = self.p.choose("What would you like to do?", [
                    "Describe what you want to hear",
                    "Choose a genre and sound settings",
                    "Describe it, then fine-tune the settings",
                    "Run the original 20-song demo",
                    "List all genres",
                    f"Preferences (results: {self.prefs.k}, LLM explanations: "
                    f"{'on' if self.prefs.use_llm else 'off'})",
                ], zero="Quit")
                if choice == -1:
                    break
                [self.describe, self.settings_only, self.describe_and_tune,
                 self.demo, self.list_genres, self.preferences][choice]()
        except Quit:
            pass
        self.p.say("\nBye!")

    # --- main menu actions --------------------------------------------------------

    def describe(self) -> None:
        query = self.ask_query()
        if query:
            self.search(query, EMPTY)

    def settings_only(self) -> None:
        flags = self.ask_settings(EMPTY)
        if flags == EMPTY:
            self.p.say("No settings chosen; nothing to search for.")
            return
        self.search("", flags)

    def describe_and_tune(self) -> None:
        query = self.ask_query()
        if not query:
            return
        self.p.say("Now fine-tune. Anything you set overrides what the description implies.")
        self.search(query, self.ask_settings(EMPTY))

    def demo(self) -> None:
        self.p.say("")
        run_demo()

    def list_genres(self) -> None:
        genres = self.catalog_genres()
        self.p.say(f"\n{len(genres)} genres:")
        width = max(len(g) for g in genres) + 2
        per_row = max(1, 78 // width)
        for i in range(0, len(genres), per_row):
            self.p.say("  " + "".join(g.ljust(width) for g in genres[i:i + per_row]))
        self.p.say("Also accepted: rap, r&b, lofi, kpop, dnb, bossa nova and other common names.")

    def preferences(self) -> None:
        while True:
            choice = self.p.choose("Preferences", [
                f"Number of results: {self.prefs.k}",
                f"LLM explanations: {'on' if self.prefs.use_llm else 'off (template reasons, instant)'}",
                f"Minimum match score: {self.prefs.min_score:.2f}",
            ], zero="Back")
            if choice == -1:
                return
            if choice == 0:
                self.prefs.k = self.ask_int("How many results (1-20)? ", 1, 20, self.prefs.k)
            elif choice == 1:
                self.prefs.use_llm = not self.prefs.use_llm
            else:
                self.prefs.min_score = self.ask_float("Minimum match score (0-1)? ", 0.0, 1.0, self.prefs.min_score)

    # --- questions ----------------------------------------------------------------

    def ask_query(self) -> str:
        self.p.say("\nDescribe the music in a few words: a genre plus some sound words works best.")
        self.p.say('  e.g. "mellow acoustic folk", "dark slow instrumental piano", "happy danceable latin"')
        self.p.say("  (Enter on its own goes back.)")
        return self.p.ask("> ")

    def ask_settings(self, current: UserProfile) -> UserProfile:
        """Walk through genre + each sound setting; '0' skips one, Enter keeps the current value."""
        profile = replace(current, favorite_genre=self.ask_genre(current.favorite_genre))
        for label, attr, levels in LEVELS:
            now = getattr(profile, attr)
            now_word = next((w for w, v in levels if v == now), None)
            if now is not None and now_word is None:
                now_word = f"{now:g}"
            options = [w for w, _ in levels]
            if attr == "target_tempo_bpm":
                options.append("enter exact BPM")
            title = f"{label}?" + (f" (current: {now_word}; Enter keeps it)" if now is not None else "")
            choice = self.choose_or_keep(title, options, keep=now is not None)
            if choice == "keep":
                continue
            if choice == -1:
                value = None
            elif attr == "target_tempo_bpm" and choice == len(levels):
                value = self.ask_float("BPM (40-220)? ", 40, 220, None)
            else:
                value = levels[choice][1]
            profile = replace(profile, **{attr: value})
        return profile

    def choose_or_keep(self, title: str, options: Sequence[str], keep: bool):
        """Like Prompter.choose with "0) skip", but an empty answer keeps the current value when keep=True."""
        self.p.say(f"\n{title}")
        for i, label in enumerate(options, 1):
            self.p.say(f"  {i}) {label}")
        self.p.say("  0) skip (don't care)")
        while True:
            answer = self.p.ask("> ")
            if answer == "" and keep:
                return "keep"
            if answer == "0":
                return -1
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.p.say(f"Please enter a number from 0 to {len(options)}" + (", or Enter to keep." if keep else "."))

    def ask_genre(self, current: Optional[str]) -> Optional[str]:
        try:
            from .rag.genres import normalize, resolve_genre
        except ImportError:
            from rag.genres import normalize, resolve_genre

        genres = self.catalog_genres()
        by_norm = {normalize(g): g for g in genres}
        keep = f" (current: {current}; Enter keeps it, 0 clears it)" if current else " (Enter to skip)"
        while True:
            typed = self.p.ask(f"\nGenre? Type a name or part of one{keep}: ")
            if not typed:
                return current
            if typed == "0":
                return None
            exact = by_norm.get(normalize(resolve_genre(typed, by_norm)))
            if exact:
                if normalize(typed) != normalize(exact):
                    self.p.say(f"Using '{exact}' for '{typed}'.")
                return exact
            matches = [g for g in genres if normalize(typed) in normalize(g)]
            if not matches:
                matches = difflib.get_close_matches(normalize(typed), list(by_norm), n=5, cutoff=0.5)
                matches = [by_norm[m] for m in matches]
            if not matches:
                self.p.say(f"No genre matches '{typed}'. Try another word, or pick 'List all genres' from the main menu.")
                continue
            if len(matches) == 1:
                self.p.say(f"Using '{matches[0]}'.")
                return matches[0]
            pick = self.p.choose(f"Genres matching '{typed}':", matches[:15], zero="none of these, type again")
            if pick != -1:
                return matches[pick]

    def ask_int(self, prompt: str, lo: int, hi: int, default: int) -> int:
        while True:
            answer = self.p.ask(prompt)
            if not answer:
                return default
            if answer.isdigit() and lo <= int(answer) <= hi:
                return int(answer)
            self.p.say(f"Please enter a whole number from {lo} to {hi}.")

    def ask_float(self, prompt: str, lo: float, hi: float, default: Optional[float]) -> Optional[float]:
        while True:
            answer = self.p.ask(prompt)
            if not answer:
                return default
            try:
                value = float(answer)
            except ValueError:
                value = None
            if value is not None and lo <= value <= hi:
                return value
            self.p.say(f"Please enter a number from {lo:g} to {hi:g}.")

    # --- searching ----------------------------------------------------------------

    def search(self, query: str, flags: UserProfile, k: Optional[int] = None) -> None:
        try:
            from .rag.generate import Explainer
        except ImportError:
            from rag.generate import Explainer

        retriever, rows = self.get_engine()
        k = k or self.prefs.k
        result = retriever.retrieve(
            query=query, flags=None if flags == EMPTY else flags, k=k, min_score=self.prefs.min_score
        )
        llm_query = query if self.prefs.use_llm else ""
        if llm_query and result.hits:
            self.p.say("Writing explanations with the local LLM (about 15 s on CPU)...")
        explanation = Explainer(rows, self.llm_client).explain(llm_query, result)
        self.p.say("")
        print_results(query, result, explanation, verbose=False, llm_requested=self.prefs.use_llm,
                      genres=[r["genres"] for r in rows], out=self.p.out)
        if 0 < len(result.hits) < k:
            self.p.say(f"\nOnly {len(result.hits)} songs reached the minimum match score "
                       f"({self.prefs.min_score:.2f}); lower it under Preferences to see more.")
        self.last = LastSearch(query, flags, result, explanation)
        self.after_results(k)

    def after_results(self, k: int) -> None:
        while True:
            choice = self.p.choose("Next?", [
                "Adjust the settings and search again",
                "Show more results",
                "Show details (profile used, notes)",
                "New search",
            ], zero="Main menu")
            if choice == -1:
                return
            if choice == 0:
                self.p.say("Settings you chose before are shown; press Enter to keep any of them.")
                return self.search(self.last.query, self.ask_settings(self.last.flags))
            if choice == 1:
                return self.search(self.last.query, self.last.flags, k=k + 5)
            if choice == 2:
                self.show_details()
            else:
                return self.new_search()

    def show_details(self) -> None:
        result, explanation = self.last.result, self.last.explanation
        self.p.say(f"\nProfile used ({result.profile_source}): {describe_profile(result.profile)}")
        if result.query_genres:
            self.p.say(f"Genres named in the description: {', '.join(result.query_genres)}")
        for n, hit in enumerate(result.hits, 1):
            dense = f"#{hit.dense_rank}" if hit.dense_rank else "-"
            feat = f"#{hit.feature_rank}" if hit.feature_rank else "-"
            self.p.say(f"  {n}. {hit.song.title}: text search {dense}, feature score {hit.feature_score:.2f} ({feat})")
        if explanation.issues:
            self.p.say("Explanation notes:")
            for issue in explanation.issues:
                self.p.say(f"  - {issue}")

    def new_search(self) -> None:
        choice = self.p.choose("New search:", [
            "Describe what you want to hear",
            "Choose a genre and sound settings",
            "Describe it, then fine-tune the settings",
        ], zero="Main menu")
        if choice != -1:
            [self.describe, self.settings_only, self.describe_and_tune][choice]()


def default_engine():
    try:
        from .rag.data import load_catalog
        from .rag.retrieve import HybridRetriever
    except ImportError:
        from rag.data import load_catalog
        from rag.retrieve import HybridRetriever
    return HybridRetriever.from_disk(), load_catalog().to_dict("records")


def run_interactive() -> int:
    try:
        InteractiveApp(Prompter()).run()
    except FileNotFoundError as e:  # index not built
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    return 0
