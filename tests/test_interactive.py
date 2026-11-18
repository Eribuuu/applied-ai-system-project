import io

import pytest
from conftest import FakeEmbedder

from src.interactive import EMPTY, InteractiveApp, Prompter
from src.rag.data import frame_from_songs
from src.rag.feature_scorer import FeatureScorer
from src.rag.generate import LLMUnavailable
from src.rag.index import make_cards
from src.rag.retrieve import HybridRetriever
from src.recommender import Recommender, load_songs


class DownClient:
    def complete(self, *a):
        raise LLMUnavailable("not running")


def small_engine():
    songs = load_songs("data/songs.csv")
    frame = frame_from_songs(songs)
    emb = FakeEmbedder()
    retriever = HybridRetriever(FeatureScorer(Recommender(songs)), emb.embed_documents(make_cards(frame)), emb,
                                dense_n=20, prf_n=5)
    return retriever, frame.to_dict("records")


def run(*answers):
    """Run the app with scripted answers; returns (app, printed text). Running out of answers quits."""
    feed = iter(answers)

    def fake_input(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    out = io.StringIO()
    app = InteractiveApp(Prompter(fake_input, out), load_engine=small_engine, llm_client=DownClient())
    app.run()
    return app, out.getvalue()


def test_quit_from_main_menu():
    app, text = run("0")
    assert "What would you like to do?" in text and "Bye!" in text
    assert app.engine is None  # nothing loaded if no search happened


def test_describe_search_shows_results():
    app, text = run("1", "calm acoustic folk", "0", "0")
    assert 'Query: "calm acoustic folk"' in text
    assert "1. " in text and "Next?" in text
    assert app.last.query == "calm acoustic folk"


def test_settings_search_uses_words_not_numbers():
    # genre rock, energy "very intense", mood "dark / sad", skip danceability/sound/tempo
    app, text = run("2", "rock", "5", "1", "0", "0", "0", "0", "0")
    flags = app.last.flags
    assert flags.favorite_genre == "rock"
    assert flags.target_energy == 0.9 and flags.target_valence == 0.1
    assert flags.target_danceability is None and flags.target_tempo_bpm is None
    assert "Feature targets only" in text
    assert app.last.result.hits[0].song.genre == "rock"


def test_genre_alias_and_partial_match():
    app, text = run("2", "rap", "0", "0", "0", "0", "3", "0", "0")
    assert "Using 'hip hop' for 'rap'." in text
    assert app.last.flags.favorite_genre == "hip hop"
    assert app.last.flags.target_tempo_bpm == 125


def test_partial_genre_offers_a_choice():
    # "pop" is a genre itself; "indie" only matches "indie pop" -> picked automatically
    app, text = run("2", "indie", "0", "0", "0", "0", "0", "0", "0")
    assert "Using 'indie pop'." in text


def test_custom_bpm():
    app, _ = run("2", "", "0", "0", "0", "0", "5", "128", "0", "0")
    assert app.last.flags.target_tempo_bpm == 128.0


def test_invalid_menu_input_reprompts():
    _, text = run("9", "abc", "0")
    assert text.count("Please enter a number from 0 to 6.") == 2


def test_no_settings_means_no_search():
    app, text = run("2", "", "0", "0", "0", "0", "0", "0")
    assert "No settings chosen" in text and app.last.result is None


def test_show_more_results_and_details():
    app, text = run("1", "upbeat pop", "2", "3", "0", "0")
    assert 5 < len(app.last.result.hits) <= 10  # default 5, then up to 5 more
    if len(app.last.result.hits) < 10:  # the 20-song catalog runs out of good matches
        assert "lower it under Preferences" in text
    assert "Profile used (pseudo-relevance" in text
    assert "text search #" in text


def test_adjust_keeps_current_settings_with_enter():
    # search: genre rock, energy very intense; then adjust: keep genre and energy (Enter), set mood happy
    answers = ["2", "rock", "5", "0", "0", "0", "0",
               "1", "", "", "5", "0", "0", "0", "0", "0"]
    app, _ = run(*answers)
    flags = app.last.flags
    assert flags.favorite_genre == "rock" and flags.target_energy == 0.9 and flags.target_valence == 0.9


def test_preferences_change_k_and_llm():
    app, text = run("6", "1", "3", "2", "0", "1", "chill jazz", "0", "0")
    assert app.prefs.k == 3 and app.prefs.use_llm is False
    assert len(app.last.result.hits) == 3
    assert "explained with template reasons" not in text  # LLM switched off on purpose: no fallback note


def test_ollama_down_note_in_interactive_mode():
    _, text = run("1", "chill jazz", "0", "0")
    assert "ollama serve" in text


def test_empty_description_goes_back():
    app, text = run("1", "", "0")
    assert app.last.result is None


def test_list_genres():
    _, text = run("5", "0")
    assert "genres:" in text and "rock" in text and "Also accepted: rap" in text


@pytest.mark.parametrize("answers", [("1",), ("2", "rock")])
def test_eof_mid_flow_quits_cleanly(answers):
    _, text = run(*answers)
    assert "Bye!" in text
