"""Commentator clients: hosted teacher (Groq), the local distilled student, and a mock."""

from __future__ import annotations

import asyncio

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: str, reasoning_effort: str = "medium", timeout: float = 60.0):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.reasoning_effort = reasoning_effort

    async def chat(self, model: str, system: str, user: str, max_tokens: int = 1500) -> tuple:
        """Returns (text, usage dict). Retries rate limits and transient errors."""
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        if "gpt-oss" in model:
            payload["reasoning_effort"] = self.reasoning_effort
        last = None
        for attempt in range(4):
            try:
                r = await self._client.post(GROQ_URL, json=payload)
            except httpx.HTTPError as e:
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after") or 2 * (attempt + 1))
                if wait > 20:
                    raise LLMError(f"rate limited for {wait:.0f}s (quota exhausted?)")
                await asyncio.sleep(wait + 0.5)
                continue
            if r.status_code >= 500:
                last = LLMError(f"HTTP {r.status_code}")
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            usage = data.get("usage") or {}
            if not text:
                last = LLMError("empty response")
                continue
            return text, usage
        raise LLMError(f"commentator failed after retries: {last}")

    async def aclose(self):
        await self._client.aclose()


class MockClient:
    """No network, no cost -- for local testing of the pipeline and UI."""

    reasoning_effort = "low"

    async def chat(self, model: str, system: str, user: str, max_tokens: int = 1500) -> tuple:
        await asyncio.sleep(0.05)
        if "Draft annotation:" in user:
            return "NO_CHANGES", {}
        import json
        rec = json.loads(user.split("JSON):\n", 1)[1].split("\n\nRecent moves", 1)[0])
        ideas = rec.get("move_ideas") or rec.get("position_context") or ["no special features"]
        return (f"[mock] {rec['san']} ({rec['quality_label']}): {rec['eval_after']['meaning']} "
                f"Key idea: {ideas[0]}."), {}

    async def aclose(self):
        pass



# ---------------------------------------------------------------- student (separate process)
# The student runs in its own spawned worker process. In the server process, torch/transformers
# are never imported next to fastapi/uvicorn (on Windows that mix segfaulted while loading
# weights, although the same load works from distill/run_student_v3.py), and a crash in the
# model can't take the web server down with it.

_worker_backend = None


def _student_init(base_model: str, adapter: str):
    global _worker_backend
    from chess_annotator.llm_backends import LocalHFBackend
    _worker_backend = LocalHFBackend(model_name=base_model, adapter_path=adapter, temperature=0.0)
    if hasattr(_worker_backend.model, "merge_and_unload"):  # fold LoRA into W once: same output, faster
        _worker_backend.model = _worker_backend.model.merge_and_unload()


def _student_ready() -> str:
    return str(_worker_backend.device)


def _student_generate(system: str, user: str, max_tokens: int) -> str:
    return _worker_backend.generate(system, user, max_tokens)


class StudentClient:
    """The distilled v3 student (Qwen2.5-1.5B-Instruct + LoRA) running on this machine.

    Uses the student prompt format it was trained on (the pipeline builds it; see
    GamePipeline._comment). One generation at a time, in move order, in a worker
    process. CPU: roughly 10-30 s per move; a CUDA GPU is used if torch sees one.
    """

    is_student = True
    reasoning_effort = "n/a"

    def __init__(self, base_model: str, adapter: str, max_tokens: int = 260):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from pathlib import Path

        path = Path(adapter)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent.parent / adapter  # relative to chess-annotator/
        if not path.exists():
            raise LLMError(f"student adapter not found: {adapter}")
        self.name = path.name
        self.max_tokens = max_tokens
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                         initializer=_student_init, initargs=(base_model, str(path)))
        try:
            self.device = self._pool.submit(_student_ready).result()  # loads the model now, not on the first move
        except Exception as e:  # noqa: BLE001
            self._pool.shutdown(wait=False, cancel_futures=True)
            raise LLMError(f"the student model failed to load in its worker process ({e!r}). "
                           "Check that `python -m distill.run_student_v3 --plies 1-2` works on its own.") from e
        self._lock = asyncio.Lock()

    async def chat(self, model: str, system: str, user: str, max_tokens: int | None = None) -> tuple:
        loop = asyncio.get_running_loop()
        async with self._lock:
            text = await loop.run_in_executor(self._pool, _student_generate, system, user,
                                              max_tokens or self.max_tokens)
        if not text:
            raise LLMError("student returned an empty comment")
        return text, {}

    async def aclose(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- student via llama.cpp (GGUF)
class LlamaServerClient:
    """The student quantized to GGUF and served by llama.cpp's llama-server (OpenAI-compatible).

    Much faster than transformers on CPU. Either connects to a llama-server you started
    (STUDENT_GGUF_URL) or starts one itself from STUDENT_GGUF and stops it on shutdown.
    """

    is_student = True
    reasoning_effort = "n/a"

    def __init__(self, url: str = "", gguf: str = "", binary: str = "llama-server", port: int = 8081,
                 threads: int = 0, max_tokens: int = 260, ctx: int = 4096):
        import atexit
        import shutil
        import subprocess
        import time
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent  # chess-annotator/
        self.proc, self.max_tokens = None, max_tokens
        if url:
            self.name = "llama-server"
        else:
            model = Path(gguf)
            if not model.is_absolute() and not model.exists():
                model = root / gguf
            if not model.exists():
                raise LLMError(f"GGUF model not found: {gguf} (see webapp/README.md, 'Faster student')")
            exe = shutil.which(binary)
            if not exe:
                for cand in (root / binary, root / "llama.cpp" / "llama-server.exe", root / "llama.cpp" / "llama-server",
                             root / "llama.cpp" / "build" / "bin" / "llama-server.exe",
                             root / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"):
                    if cand.exists():
                        exe = str(cand)
                        break
            if not exe:  # the prebuilt Windows zip unpacked outside the project (see README)
                for d in (Path.home() / "tools" / "llama.cpp", root / "llama.cpp" / "bin"):
                    hits = sorted(d.rglob("llama-server.exe")) + sorted(d.rglob("llama-server")) if d.exists() else []
                    if hits:
                        exe = str(hits[0])
                        break
            if not exe:
                raise LLMError(f"llama-server not found ({binary!r}); set LLAMA_SERVER to the full path of "
                               "llama-server.exe, e.g. export LLAMA_SERVER=~/tools/llama.cpp/bin/llama-server.exe")
            cmd = [exe, "-m", str(model), "--jinja", "-c", str(ctx), "-np", "1",
                   "--host", "127.0.0.1", "--port", str(port)]
            if threads > 0:
                cmd += ["-t", str(threads)]
            self._log_path = root / "llama_server.log"
            self._log = open(self._log_path, "w", encoding="utf-8")
            print("Starting:", " ".join(cmd), f"(log: {self._log_path.name})")
            self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
            atexit.register(self._stop)
            url = f"http://127.0.0.1:{port}"
            self.name = model.name
        self.url = url.rstrip("/")

        deadline = time.time() + 300
        while True:  # /health answers 503 while the model loads, 200 when ready
            if self.proc and self.proc.poll() is not None:
                tail = self._log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise LLMError(f"llama-server exited with code {self.proc.returncode}:\n{tail}")
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                self._stop()
                raise LLMError(f"llama-server at {self.url} did not become ready")
            time.sleep(0.5)
        self.device = f"llama.cpp ({self.name})"
        self._client = httpx.AsyncClient(timeout=600)
        self._lock = asyncio.Lock()

    async def chat(self, model: str, system: str, user: str, max_tokens: int | None = None) -> tuple:
        payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": 0, "max_tokens": max_tokens or self.max_tokens, "cache_prompt": True}
        async with self._lock:  # one at a time, so comments arrive in move order
            r = await self._client.post(self.url + "/v1/chat/completions", json=payload)
        if r.status_code >= 400:
            raise LLMError(f"llama-server HTTP {r.status_code}: {r.text[:200]}")
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise LLMError("student returned an empty comment")
        return text, {}

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.proc.kill()

    async def aclose(self):
        await self._client.aclose()
        self._stop()
