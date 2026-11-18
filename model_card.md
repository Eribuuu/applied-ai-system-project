# 🎧 Model Card: Music Recommender + RAG

**What it is:** a music recommender that takes a plain-language request, retrieves songs from an 81,000-track Spotify catalog with hybrid search (embeddings + the original feature scorer, fused with RRF), and has a local LLM (`qwen2.5:3b` via Ollama) explain the picks. A guardrail checks every explanation and falls back to template reasons. Details are in the [README](README.md).

---

## Limitations and biases

**Data**
- **Genre labels are Spotify's, not listeners'.** There is no "lofi" genre, and Disney sits right next to piano because both are quiet and acoustic. The system inherits these choices.
- **Uneven coverage.** Each of the 114 genres started at about 1,000 tracks, but after deduplication some have far fewer, and there is no mood, era, lyric or language data at all.
- **Deduplication favors popular versions.** When the same song appears more than once, the most popular copy is kept, which nudges results toward mainstream releases.

**Retrieval**
- **English-only embedding model.** `bge-small-en` handles English queries well and other languages poorly, even though the catalog has Turkish, Indian, Brazilian and K-pop music.
- **Title words can outweigh meaning.** "angry heavy metal for the gym" once ranked a comedy track called *Gym* in second place.
- **Descriptor words use hard cut-offs.** Energy 0.59 is "moderate energy" and 0.60 is "energetic", so near-identical songs can be described differently.
- **Genre centroids only see audio features.** By those features hip-hop sits closest to j-dance and dance, so a hip-hop request gives those genres partial credit.
- **No query understanding.** Negation ("nothing acoustic"), numbers in words ("120 bpm") and requests naming two genres are not understood.

**Generation**
- **The guardrail checks IDs and numbers, not every claim.** qwen once said "all songs are hip-hop" when one was electronic, and that got through.
- **Explanations are repetitive.** Sometimes all five songs get the same sentence.

**Evaluation**
- **The eval is small and subjective.** It uses 10 queries whose relevance criteria I wrote, so the 0.75 score reflects my idea of a good match.

---

## Could it be misused, and how would I prevent that?

The risk is low: the system only recommends existing songs and explains them. Some misuse is still possible:

| Risk | What's in place | What I'd add |
|---|---|---|
| **The LLM inventing songs or facts** | The LLM never picks songs. The guardrail rejects unknown song IDs and numbers that aren't in the context, and falls back to template text. | A check that genre claims match the song's real genres |
| **Prompt injection** through a query or a song title ("ignore your rules…") | Output is forced into a fixed JSON format, and only text that passes the guardrail is shown. A bad reply can only produce template text. | Strip instruction-like text from titles before building the context |
| **Explicit songs reaching children** ("cheerful songs for kids") | Nothing yet. The `explicit` flag is loaded but not used for filtering. | A "no explicit lyrics" option, on by default for kid-related queries |
| **Rigging rankings** (e.g. editing weights or aliases to push certain artists) | All weights and aliases are in plain code and covered by tests | Log changes to scoring settings; show "Show details" by default |
| **Privacy, if switched to a cloud LLM** | The local LLM keeps everything on the machine | Warn the user and document what data is sent if a hosted API is used |

---

## What surprised me while testing reliability

- **My first explanation of a problem was wrong.** The piano query dropped from 1.0 to 0.8, so I assumed the genre weight was too strong and lowered it. Every query got worse, including piano. The real cause was that the label didn't count calm Disney instrumentals as relevant, even though the genre centroids rank Disney closest to piano.
- **Small models misread numbers.** One model turned "tempo within 14 BPM of target" into "a target tempo of 14 BPM". The number was real, so my check passed it, but the meaning was wrong. Rewording the facts as "tempo 152 BPM (target 150)" fixed that kind of mistake.
- **Valid output doesn't mean useful output.** All four LLMs I compared returned valid JSON with real song IDs every time. The differences were behaviour: two skipped songs without saying so, and one described weak matches as great fits.
- **An improvement didn't show up in the score.** Adding genre aliases to the song cards put 9 hip-hop songs in the top 10 for "rap", but the eval score didn't move, because the eval's rap query was already fixed by genre detection.
- **The guardrail never had to reject anything in live runs.** All 15 live explanations passed, yet one of them still contained a false genre claim. Passing the guardrail is not the same as being correct.

---

## Collaboration with AI

I built this with Claude Code as a coding assistant. I worked one step at a time: it proposed a plan, implemented one step, and ran the tests, and I reviewed each step before moving on. I made the design calls: the dataset, local vs. cloud LLM, menu style, and when to stop tuning.

**A suggestion I accepted:** when the rap query scored 0 out of 10, Claude found that the real problem was vocabulary, not the scoring. The word "rap" never maps to Spotify's "hip-hop", so the existing genre centroids never got used. It suggested an alias table plus detecting a single genre named in the query. I accepted it because it was small, testable and needed no index rebuild. It raised hybrid precision from 0.63 to 0.75 and took the rap query from 0.0 to 0.5.

**Suggestions I didn't accept as written:**
- **The first architecture was over-engineered for a proof of concept.** Claude's first plan had three retrievers (adding BM25), diversity reranking, genre knowledge notes, a Streamlit app, run logs and an LLM query parser. I scoped it down to dense search plus the existing scorer, a CLI and one LLM call. That was enough to show RAG working and keep it testable, and the rest went under future work.
- **I changed the LLM provider.** The plan used the Claude API for explanations. I switched to a small local model because I wanted it free, offline and private. Claude then compared four local models, and I chose `qwen2.5:3b` from the results.
- **I chose simpler menus.** Claude recommended arrow-key menus using an extra library. I chose plain numbered menus so the app works in any terminal with no new dependency.
- **A flawed suggestion, caught by measurement.** Claude's fix for the piano query (lower the genre weight) was based on a wrong guess about the cause. Because we ran the eval at four weights before keeping it, the regression was obvious, the change was reverted, and the real cause turned out to be the label.
