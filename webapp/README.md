# Chess commentary web app

Paste a PGN, get move-by-move commentary. Stockfish computes the evaluations,
`chess_annotator/positional.py` finds the human-style ideas (open files, pins,
forks, pawn structure, endgame type...), and a hosted model on Groq writes the
prose from those verified facts. An optional second pass fact-checks each
comment against the same facts.

## Run locally (Git Bash, from `chess-annotator/`)

```bash
pip install -r requirements.txt -r requirements-app.txt
export GROQ_API_KEY="gsk_..."
export STOCKFISH_PATH="./stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
uvicorn webapp.server:app --reload
```

Student model instead of Groq (needs `pip install torch transformers accelerate peft`; no API key, no cost, no fact-check pass):

```bash
export COMMENTATOR_BACKEND=student
python -m uvicorn webapp.server:app   # no --reload; the model loads in a separate worker process
```

Free test run with no API calls: `export COMMENTATOR_BACKEND=mock` before starting.

In a second terminal:

```bash
python -m webapp.try_api examples/sample_game.pgn
```

## Frontend

`static/index.html` is a single self-contained page (no build step), served at `/`.
Open http://127.0.0.1:8000 after starting the server.

- Load a PGN by dropping a file, pasting, or using the built-in sample (Morphy’s Opera Game).
  Multi-game files show a filterable picker; only the chosen game is sent to the API.
- The board is replayed in the browser from the PGN, so you can step through the game
  immediately. Engine facts and comments from `/api/annotate` attach to each ply as they stream in.
- Playback (Space) can hold on each move until its comment arrives ("Wait for commentary").
- The PGN's own comments, NAGs and `[%cal]`/`[%csl]` arrows/highlights are shown alongside
  the generated commentary; the engine's preferred move is drawn as a dashed arrow.
- Eval bar and eval graph (click to jump), captured material, flip (F), arrow keys, Home/End.
- "Export annotated PGN" downloads the game with the generated comments and `[%eval]` tags.
- If the server is unreachable the page still works as a PGN viewer.
  Use `?api=http://host:port` to point the page at a different server (that server needs CORS).

## Faster student (llama.cpp, CPU)

`COMMENTATOR_BACKEND=student` runs the student through transformers in fp32, the slowest
option on CPU. `student_gguf` runs the same model quantized with llama.cpp instead.

One-time setup (Git Bash, from `chess-annotator/`):

```bash
# 1. Fold the LoRA into the base weights -> student-v3-merged/
python -m distill.export_student_gguf

# 2. llama.cpp: the repo for the convert script, the prebuilt Windows CPU binaries for the rest.
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install sentencepiece protobuf
#    Download llama-<version>-bin-win-cpu-x64.zip from https://github.com/ggml-org/llama.cpp/releases
#    and unzip it into llama.cpp/bin/

# 3. Convert, then quantize
python llama.cpp/convert_hf_to_gguf.py student-v3-merged --outtype bf16 --outfile student-v3-bf16.gguf
./llama.cpp/bin/llama-quantize.exe student-v3-bf16.gguf student-v3-q8_0.gguf Q8_0
./llama.cpp/bin/llama-quantize.exe student-v3-bf16.gguf student-v3-q4_k_m.gguf Q4_K_M   # optional
```

Check quality and speed before switching (second terminal runs the server):

```bash
./llama.cpp/bin/llama-server.exe -m student-v3-q8_0.gguf --jinja -c 4096 --port 8081
python -m distill.eval_gguf --n 150 --out distill/data/v3/gguf_q8_eval.json
```

It compares consistency-check pass rates with the original student on the same test records
and prints seconds per comment and tokens/s. Q8_0 should match the original closely; check Q4_K_M
the same way before using it.

Run the app (it starts and stops llama-server itself):

```bash
export COMMENTATOR_BACKEND=student_gguf
export STUDENT_GGUF=student-v3-q8_0.gguf
export LLAMA_SERVER=./llama.cpp/bin/llama-server.exe
python -m uvicorn webapp.server:app --port 8000
```

`STUDENT_GGUF_URL=http://127.0.0.1:8081` uses a llama-server you already started instead.
`STUDENT_THREADS` sets llama.cpp's thread count (default: its own choice); leaving a core free
for Stockfish can help.

## API

`POST /api/annotate` with `{"pgn": "...", "factcheck": true}` returns a
newline-delimited JSON stream of events: `meta`, then `move` events (engine
facts + insights, as soon as Stockfish finishes each move) and `comment`
events (as each comment is written, possibly out of order), then `done`.
Errors before streaming come back as HTTP 400/429/503 with `{"error": "..."}`.

`GET /api/health` shows the model and today's spend.

## Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | Groq key (server-side only, never sent to browsers) |
| `COMMENTATOR_BACKEND` | `groq` | `student` = distilled v3 student run locally (transformers); `student_gguf` = same, quantized via llama.cpp (faster); `mock` for free local testing |
| `STUDENT_BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | student base model (downloaded from Hugging Face on first run) |
| `STUDENT_ADAPTER` | `chess-annotator-lora-v3` | LoRA adapter folder (relative to `chess-annotator/`) |
| `STUDENT_MAX_TOKENS` | `260` | max tokens per student comment |
| `STUDENT_GGUF` / `STUDENT_GGUF_URL` | `student-v3-q8_0.gguf` / – | `student_gguf` backend: model file, or an already-running llama-server |
| `LLAMA_SERVER` / `LLAMA_PORT` / `STUDENT_THREADS` | `llama-server` / `8081` / `0` | llama-server binary, its port, CPU threads (0 = default) |
| `COMMENTATOR_MODEL` | `openai/gpt-oss-120b` | model that writes the commentary |
| `REASONING_EFFORT` | `medium` | gpt-oss reasoning effort |
| `FACTCHECK_DEFAULT` | `true` | run the fact-check pass unless the request says otherwise |
| `FACTCHECK_MODEL` | `openai/gpt-oss-120b` | model for the fact-check pass |
| `STOCKFISH_PATH` | `stockfish` | engine binary |
| `ENGINE_DEPTH` | `14` | search depth per position |
| `MAX_PLIES` / `MAX_PGN_CHARS` | `200` / `30000` | reject longer games |
| `MAX_CONCURRENT_GAMES` | `2` | games analysed at once (CPU bound) |
| `LLM_CONCURRENCY` | `6` | commentary calls in flight across all games |
| `GAMES_PER_IP_PER_HOUR` | `10` | per-visitor rate limit |
| `DAILY_BUDGET_USD` | `2.0` | stop commentary for the day once spent |
| `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` | built-in table | override model prices (USD per million tokens) |
| `DB_PATH` | `webapp_data.sqlite3` | cache of finished games + spend ledger |
| `TRUST_PROXY_HEADERS` | `false` | use `X-Forwarded-For` for rate limiting (set `true` behind a proxy) |

Finished games are cached by (moves, settings, pipeline version), so the same
game requested twice costs nothing the second time.
