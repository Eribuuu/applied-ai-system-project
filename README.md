# 🎵 Music Recommender + RAG

Describe the music you want in plain words ("mellow acoustic folk") and get songs from an 81,000-track Spotify catalog, each with a short explanation written by a local LLM that is only allowed to use facts about the retrieved songs.

## Original project

This extends my earlier **Music Recommender Simulation**. That project was a content-based recommender: it scored a 20-song catalog against a hand-written taste profile (target genre, energy, valence, danceability, tempo, acousticness) using Gaussian closeness and a weighted average, then explained each pick with a "Because: …" line. Its goal was to show how a recommender turns data into ranked predictions and where its weights and labels create bias. The original write-up, experiments and outputs are in [docs/original_project.md](docs/original_project.md).

## Title and summary

**Music Recommender + RAG** adds retrieval-augmented generation on top of the original scorer:

- **Retrieve:** a hybrid retriever finds songs by meaning (embedding search over text "cards" describing each song) and by audio features (the original scoring math, vectorized for 81k songs), then merges both rankings.
- **Augment:** the top songs and their real feature values become the context for the LLM.
- **Generate:** `qwen2.5:3b`, running locally through Ollama, writes the explanations. A guardrail rejects anything ungrounded and falls back to the original template reasons.

Why it matters: people describe music in words, not as energy 0.73. The project shows how to connect that kind of request to a structured catalog without letting the LLM pick songs or make up facts.

## Architecture overview

Diagram: [diagrams/system_architecture.mmd](diagrams/system_architecture.mmd) (paste into [mermaid.live](https://mermaid.live) to view).

1. **Offline indexing (once):** the Kaggle CSV (114k rows) is deduped to 81k songs. Each song gets a card such as *"Breaker by Accept. Genres: heavy metal. Sound: high-energy, intense, …"*, which is embedded with `bge-small` and saved to `data/index/`.
2. **Input:** interactive numbered menus, or `--query` and feature flags.
3. **Retriever** (no LLM):
    - **Dense search:** the top 300 cards closest to the query.
    - **Profile:** your settings first, then a genre named in the query ("rap" → hip-hop), then the average features of the top 20 dense hits for anything left unset.
    - **Scoring and fusion:** the original feature scorer ranks all songs against that profile, and **Reciprocal Rank Fusion** merges the two rankings.
4. **Generator:** a context builder states each song's facts ("energy 0.31 (target 0.30)"). qwen2.5:3b replies in a fixed JSON format, and the **guardrail** checks it: song IDs must be real, numbers must come from the context, and every song must be explained.
5. **Output:** ranked songs with match scores and explanations. "Show details" displays the profile used and each song's rank in both retrievers.
6. **Testing and people:** pytest, a retrieval evaluator on hand-labeled queries, and an LLM comparison script. I wrote the labels, tuned the system with the eval, picked the model and spot-check explanations.

## Setup

Requires Python 3.10+, about 4 GB of free RAM, and about 3 GB of disk space (model plus index).

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Data: download the Kaggle "Spotify Tracks Dataset" (114k tracks, 114 genres)
#    and save it as data/dataset.csv

# 3. Build the catalog and the embedding index (once; ~12 min on CPU)
python -m src.rag.data
python -m src.rag.index

# 4. Local LLM: install Ollama (https://ollama.com/download), then
ollama pull qwen2.5:3b
ollama serve            # leave running in another terminal

# 5. Run
python src/main.py      # interactive menus
pytest                  # tests
```

Other ways to run it:
- `python src/main.py -q "dark slow piano" --energy 0.2` for a one-off search
- `--no-llm` to skip the LLM
- `--demo` for the original 20-song demo
- `python scripts/eval_retrieval.py` for the retrieval eval

Without Ollama running, search still works and explanations fall back to template reasons.

## Sample interactions

**1. Describing the music (menu option 1)**
```
> mellow acoustic folk
The picks are all folk songs with a mellow, acoustic sound.

1. Warm with You — Hayden Calnin  [folk]  match 0.95
   Matches the mellow, acoustic folk request with its calm, melancholic instrumental sound.
2. Hansnuhana — Fossils  [folk, indian, indie]  match 0.94
   While not exclusively folk, it fits the acoustic and mellow request with its moderate energy and upbeat tempo.
3. Blessings — Hollow Coves  [folk]  match 0.92
   A solid fit for the mellow, acoustic folk request with its moderate energy and acoustic sound.
```

**2. A request with no matching genre name**

Spotify has no "lofi" genre. The query still finds calm `study` tracks through the embeddings.
```
$ python src/main.py -q "mellow songs for late-night studying" -k 3
All songs are selected for their calm, mellow, and melancholic sounds suitable for late-night studying.

1. morning coffe — inbetweenjobs  [study]  match 0.98
   Matches the listener's request with its calm, melancholic, and slow tempo, ideal for late-night studying sessions.
2. Sleepover — Timothy Infinite  [study]  match 0.96
   Completely aligns with the listener's request, being instrumental, calm, and mellow, perfect for a quiet study session.
3. Night Sky — Faime  [chill]  match 0.95
   While not a direct match for study, its calm and melancholic sound complements the request, offering a chill alternative to the other songs.
```

**3. Settings only (menu option 2): genre "rap", energy "energetic"**

No text is given, so there's no LLM call and the original reasons are used. "rap" resolves to hip-hop.
```
Genre? rap
Using 'hip-hop' for 'rap'.
...
1. When in Rome — Mac Miller  [hip-hop]  match 1.00
   Because: same genre (hip-hop), energy within 0.00 of target
2. Ek Din Pyaar — MC STAN  [hip-hop]  match 1.00
   Because: same genre (hip-hop), energy within 0.00 of target
```

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| **Hybrid retrieval** (dense search + original scorer, fused with RRF) | Embeddings understand "mellow" but not numbers. The scorer is precise and explainable but needs targets. Each covers the other's gap. | More moving parts. RRF ignores score sizes and uses only positions. |
| **Songs described as words on the cards** ("calm, mellow"), never numbers | Embedding models don't understand that 0.35 is close to 0.36 | Binning loses detail; the numeric fit is left to the scorer |
| **Profile from the average of the top 20 dense hits** | Lets a text-only query drive the numeric scorer with no LLM parser | Inherits whatever the dense search got wrong |
| **Genre alias table** (rap → hip-hop, lofi → study) | Neither the embeddings nor the genre centroids map user words to Spotify's names | Hand-written and incomplete |
| **The LLM only explains**; retrieval picks and ranks the songs | Keeps ranking deterministic and testable, and the LLM can't recommend songs that don't exist | Explanations can't fix bad retrieval |
| **Local `qwen2.5:3b`** via Ollama, chosen by comparing 4 models | Free, private, offline. It was the only tested model that explained every song *and* flagged weak matches honestly. | ~15 s per answer on CPU; prose is repetitive |
| **Guardrail plus template fallback** | Output is always either verified LLM text or the original reasons | It checks IDs and numbers, not every claim |
| **No LLM query parser yet** | Keeps the pipeline deterministic and easy to test first | Negation ("nothing acoustic") and numbers in words aren't understood. Use the settings for those. |
| **Numbered menus, no extra libraries** | Works in any terminal | Plainer than arrow-key menus |

## Testing summary

**Automated tests:** `pytest` runs 137 tests, all passing; 1 live-LLM test is skipped unless `RUN_OLLAMA_TESTS=1` is set. They cover:
- data cleaning
- the vectorized scorer's scores matching the original `Recommender` within 1e-12 and giving the same rankings
- song cards, RRF and genre aliases
- the guardrail, with a fake LLM returning bad JSON, unknown IDs, invented numbers and skipped songs
- the CLI and menus, using scripted input

**Retrieval eval** (`scripts/eval_retrieval.py`, precision@10 on 10 hand-labeled queries):

| | Dense only | Features only | Hybrid |
|---|---|---|---|
| First version | 0.61 | 0.60 | 0.63 |
| + genre detection and aliases | 0.61 | 0.69 | **0.75** |

**LLM comparison** (`scripts/compare_llms.py`, 5 queries each): all four models returned valid JSON with real IDs.
- `llama3.2:3b` and `qwen2.5:1.5b` silently skipped songs.
- `llama3.2:1b` covered every song but presented weak matches as good ones.
- `qwen2.5:3b` covered everything and said when a pick fit poorly.

**What worked:**
- The hybrid approach beat both retrievers on their own.
- Detecting the genre in the query fixed the worst queries: rap went from 0.0 to 0.5, and metal from 0.5 to 0.9.
- The guardrail has never let an invented number or song through.

**What didn't:**
- Lowering the genre weight to fix one piano query made *every* query worse. It turned out the "misses" were calm Disney instrumentals, which the genre centroids place next to piano, so the label was the real issue.
- Adding aliases to the cards helped short queries ("rap": 9 of the top 10 are hip-hop) but not long ones, where other words dominate.
- The LLM still makes false claims without numbers, such as "all songs are hip-hop", that the guardrail can't catch.

**What I learned:**
- Measure before tuning. My first explanation of the piano drop was wrong, and the eval showed it.
- Small LLMs are more reliable when they get unambiguous facts ("value (target X)") and a fixed output format.
- An eval is only as good as its labels, and writing them is a human judgment call.

**Refelction:**
- What I learned with working with AI is that it is a very powerful tool but needs to have the precise amount of information and guidance to achieve the goals you are trying to achieve. Its is a good collaborator and I was able to bounce ideas off of it and shape what my actual vision was for this project.

More detail: [model_card.md](model_card.md) · original project: [docs/original_project.md](docs/original_project.md)
