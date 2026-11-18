"""
Generation: a small local LLM explains the songs retrieval already picked.

The LLM never chooses or reorders songs; it only writes the explanations. Every
reply goes through a guardrail, and anything that fails it is replaced by the
Recommender's template reasons, so the output is always grounded:

  - The reply must be JSON matching SCHEMA; otherwise every song gets template text.
  - Each pick must cite a retrieved id (S1..Sk). Unknown or repeated ids are ignored.
  - Any number in the text must appear in the context (catches made-up stats).
  - Every retrieved song gets an explanation: songs the LLM skipped or whose
    text failed a check get template text instead.
  - If the LLM can't be reached (Ollama not running), all songs get template text.

The context states each feature as "value (target X)" rather than the
Recommender's "within X of target" phrasing, which small models misread
(one called "within 14 BPM" a "target tempo of 14 BPM").

Model: qwen2.5:3b through Ollama (http://localhost:11434), chosen with
scripts/compare_llms.py. Start the server with `ollama serve`.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .cards import descriptors
from .retrieve import Hit, RetrievalResult

try:
    from ..recommender import Recommender
except ImportError:  # run as `python src/main.py`, where src/ is the top level
    from recommender import Recommender

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You explain music recommendations. You get a listener's request and a numbered list of "
    "songs a recommender already picked, with facts about each.\n"
    "Rules:\n"
    "- Use only the facts given. Do not invent numbers, years, artists, lyrics or other details.\n"
    "- Only discuss the listed songs, referring to each by its id (S1, S2, ...).\n"
    "- For every song, write one or two sentences on why it fits the request. "
    "If a song fits poorly, say so honestly.\n"
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

# Feature name -> (UserProfile attr, Song attr, format)
SHOWN_FEATURES = {
    "energy": ("target_energy", "energy", "{:.2f}"),
    "valence": ("target_valence", "valence", "{:.2f}"),
    "danceability": ("target_danceability", "danceability", "{:.2f}"),
    "acousticness": ("target_acousticness", "acousticness", "{:.2f}"),
    "tempo": ("target_tempo_bpm", "tempo_bpm", "{:.0f} BPM"),
}

NUMBER = re.compile(r"\d+(?:\.\d+)?")
SONG_ID = re.compile(r"\bS(\d+)\b")


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> str: ...


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    """Chat with a local Ollama model, its output constrained to a JSON schema."""

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL, timeout: float = 120.0):
        self.model, self.url, self.timeout = model, url, timeout

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        req = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            raise LLMUnavailable(f"could not get a reply from {self.model} at {self.url}: {e}") from e


@dataclass
class ExplainedSong:
    hit: Hit
    text: str
    source: str  # "llm" or "template"


@dataclass
class Explanation:
    summary: Optional[str]          # None when the LLM's summary failed a check or there was no LLM reply
    songs: List[ExplainedSong]      # same order as the retrieval hits
    used_llm: bool                  # True if any text came from the LLM
    issues: List[str] = field(default_factory=list)  # what the guardrail caught, for logs and tests


def template_text(hit: Hit) -> str:
    return "Because: " + ", ".join(hit.reasons)


def _fmt_target(fmt: str, value: Optional[float]) -> str:
    return "" if value is None else f" (target {fmt.format(value)})"


def build_context(query: str, result: RetrievalResult, rows: Sequence[Mapping[str, Any]]) -> str:
    """The user message: the request plus numbered facts about each retrieved song."""
    profile = result.profile
    lines = [f'Listener request: "{query}"', "", "Retrieved songs:"]
    for n, hit in enumerate(result.hits, 1):
        row, song = rows[hit.index], hit.song
        audio = [
            f"{name} {fmt.format(getattr(song, song_attr))}{_fmt_target(fmt, getattr(profile, user_attr))}"
            for name, (user_attr, song_attr, fmt) in SHOWN_FEATURES.items()
        ]
        genre_note = f" (target genre {profile.favorite_genre})" if profile.favorite_genre else ""
        lines.append(
            f'[S{n}] "{song.title}" by {song.artist}. '
            f'Genres: {", ".join(row["genres"])}{genre_note}. '
            f'Sound: {", ".join(descriptors(row))}. '
            f'Audio: {", ".join(audio)}. '
            f"Overall match {hit.feature_score:.2f} out of 1."
        )
    return "\n".join(lines)


def _numbers(text: str) -> set:
    return {float(n) for n in NUMBER.findall(text)}


def _grounded(text: str, known_numbers: set, k: int) -> Optional[str]:
    """None if the text is fine, else what's wrong with it."""
    bad_ids = sorted({f"S{n}" for n in SONG_ID.findall(text) if not 1 <= int(n) <= k})
    if bad_ids:
        return f"cites songs that weren't retrieved: {', '.join(bad_ids)}"
    invented = sorted(_numbers(text) - known_numbers)
    if invented:
        return f"numbers not in the context: {', '.join(f'{x:g}' for x in invented)}"
    return None


def apply_guardrail(reply: str, result: RetrievalResult, context: str) -> Explanation:
    """Turn the raw LLM reply into an Explanation, replacing anything ungrounded with template text."""
    hits, k = result.hits, len(result.hits)
    issues: List[str] = []
    try:
        data = json.loads(reply)
        summary, picks = data["summary"], data["picks"]
        if not isinstance(summary, str) or not isinstance(picks, list):
            raise TypeError("summary must be a string and picks a list")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        issues.append(f"reply is not the expected JSON ({e}); used template text for every song")
        return Explanation(None, [ExplainedSong(h, template_text(h), "template") for h in hits], False, issues)

    known = _numbers(context) | {float(i) for i in range(1, k + 1)}
    texts: Dict[int, str] = {}
    for pick in picks:
        if not isinstance(pick, dict):
            issues.append(f"ignored malformed pick {pick!r}")
            continue
        sid, why = str(pick.get("id", "")).strip(), pick.get("why")
        m = re.fullmatch(r"S(\d+)", sid)
        if not m or not 1 <= int(m.group(1)) <= k:
            issues.append(f"ignored pick for unknown song id {sid!r}")
            continue
        n = int(m.group(1))
        if n in texts:
            issues.append(f"ignored repeated pick for {sid}")
            continue
        if not isinstance(why, str) or not why.strip():
            issues.append(f"{sid} has no explanation; used template text")
            continue
        problem = _grounded(why, known, k)
        if problem:
            issues.append(f"{sid} {problem}; used template text")
            continue
        texts[n] = why.strip()

    songs = []
    for n, hit in enumerate(hits, 1):
        if n in texts:
            songs.append(ExplainedSong(hit, texts[n], "llm"))
        else:
            if not any(i.startswith(f"S{n} ") for i in issues):
                issues.append(f"S{n} was not explained by the LLM; used template text")
            songs.append(ExplainedSong(hit, template_text(hit), "template"))

    summary_problem = _grounded(summary, known, k) if summary.strip() else "is empty"
    if summary_problem:
        issues.append(f"summary {summary_problem}; dropped it")
        summary = None
    else:
        summary = summary.strip()

    used = summary is not None or any(s.source == "llm" for s in songs)
    return Explanation(summary, songs, used, issues)


class Explainer:
    def __init__(self, rows: Sequence[Mapping[str, Any]], client: Optional[LLMClient] = None):
        """
        rows: catalog rows (dicts) in catalog order, for genres and descriptor words.
        client: the LLM; defaults to qwen2.5:3b on the local Ollama server.
        """
        self.rows = rows
        self.client = client or OllamaClient()

    def explain(self, query: str, result: RetrievalResult) -> Explanation:
        hits = result.hits
        if not hits:
            return Explanation(None, [], False, ["no songs to explain"])
        if not query.strip():
            # Flags only: there's no request in words to relate the songs to.
            return Explanation(None, [ExplainedSong(h, template_text(h), "template") for h in hits], False,
                               ["no text query; used template text"])
        context = build_context(query, result, self.rows)
        try:
            reply = self.client.complete(SYSTEM_PROMPT, context, SCHEMA)
        except LLMUnavailable as e:
            return Explanation(None, [ExplainedSong(h, template_text(h), "template") for h in hits], False,
                               [f"LLM unavailable ({e}); used template text"])
        return apply_guardrail(reply, result, context)
