"""Vision-language scoring through llama.cpp's own server: one child process per Rizzo server.

The in-process ctypes path (`backend_llama`) feeds token ids at positions 0, 1, 2, …; image
embeddings from libmtmd need model-specific positions (M-RoPE for Qwen-VL) that it does not
model. The pinned runtime ships `llama-server`, which does, so this backend starts it on a
private port and keeps what makes the answers Rizzo's: the prompt, single-token answer letters
and the softmax over their logits (`decisions.decode`).

Letter logits come from `/completion` with `n_predict: 1` and `n_probs`. Without
`post_sampling_probs`, llama-server reports the log-softmax of the raw logits over the whole
vocabulary (`get_token_probabilities`, release b11081): every logit minus one constant, which
the softmax over the letters cancels. Nothing is decoded after the prompt.
"""

import atexit
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import llama_release
from .backend_llama import compile_template
from .config import VISION
from .prompts import MEDIA_MARKER, PROMPT_VERSION, Compiled, canonical

SERVER = "llama-server"
N_PROBS = 256  # top tokens reported; an offered letter below them gets the lowest reported one
IMAGE_CELLS = 2048  # context cells beyond --ctx, which counts the text tokens only
STARTUP_SECONDS = 300
REQUEST_SECONDS = 120
LOG_FILE = Path("llama-server.log")
LOG_ENV = "RIZZO_LLAMA_LOG"
RESPONSE_FIELDS = ["completion_probabilities", "tokens_evaluated", "timings", "truncated"]


class ServerClient:
    """JSON over HTTP to the child server; failures become ValueError (HTTP 422 upstream)."""

    def __init__(self, base_url: str, timeout: float = REQUEST_SECONDS):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, path: str) -> dict:
        return self._call(urllib.request.Request(self.base_url + path))

    def post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        return self._call(urllib.request.Request(self.base_url + path, data, headers))

    def _call(self, request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise ValueError(
                f"{SERVER} {request.full_url} → HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise ValueError(f"{SERVER} at {self.base_url} is not answering: {error}") from error


class LlamaServer:
    """The child process: started with the Rizzo server, stopped with it."""

    def __init__(self, process, client, log_path):
        self.process = process
        self.client = client
        self.log_path = log_path

    @staticmethod
    def command(binary, model, projector, port, n_ctx, device="auto", threads=None) -> list[str]:
        on_cpu = device == "cpu"
        command = [
            str(binary),
            "--model", str(model),
            "--mmproj", str(projector),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(n_ctx),
            "--n-gpu-layers", "0" if on_cpu else "999",
            "--parallel", "1",  # Rizzo serializes requests already
            "--no-webui",
        ]  # fmt: skip
        if on_cpu:
            command.append("--no-mmproj-offload")
        if threads:
            command += ["--threads", str(threads)]
        return command

    @classmethod
    def start(cls, command_for_port, log_path=LOG_FILE, timeout=STARTUP_SECONDS):
        port = free_port()
        verbose = os.environ.get(LOG_ENV) == "1"
        with contextlib.ExitStack() as stack:  # the child keeps its own descriptor of the log
            log = None if verbose else stack.enter_context(Path(log_path).open("ab"))
            process = subprocess.Popen(
                command_for_port(port),
                stdout=log,
                stderr=subprocess.STDOUT if log else None,
                stdin=subprocess.DEVNULL,
            )
        server = cls(process, ServerClient(f"http://127.0.0.1:{port}"), log_path)
        atexit.register(server.close)  # also when the Rizzo server stops on an error
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise ValueError(
                    f"{SERVER} exited with code {process.returncode} while loading; "
                    f"see {'its output above' if verbose else Path(log_path).resolve()}"
                )
            try:
                server.client.get("/health")
                return server
            except ValueError:  # 503 while the model loads, refused before it listens
                if time.monotonic() > deadline:
                    server.close()
                    raise ValueError(f"{SERVER} did not become ready in {timeout} s") from None
                time.sleep(0.25)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerTokenizer:
    """`prompts.compile_request`'s tokenizer calls, answered by the child server: the GGUF chat
    template rendered here as for Spark, the vocabulary through `/tokenize`."""

    pad_token_id = None
    eos_token_id = None

    def __init__(self, client, template: str):
        self.client = client
        self.template = compile_template(template)
        self._short = {}  # answer letters are encoded on every request

    def apply_chat_template(self, messages, tokenize=False, **variables) -> str:
        if tokenize:
            raise ValueError("Render the text, then call encode()")
        return self.template.render(messages=messages, **variables)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        key = (text, add_special_tokens)
        if key in self._short:
            return self._short[key]
        body = {"content": text, "add_special": add_special_tokens, "parse_special": True}
        tokens = self.client.post("/tokenize", body)["tokens"]
        if len(text) <= 4:
            self._short[key] = tokens
        return tokens


class VisionBackend:
    """Same contract as `LlamaBackend` (tokenizer, `metadata`, `score`), plus images."""

    vision = True

    def __init__(self, server, tokenizer, metadata, marker=MEDIA_MARKER):
        self.server = server
        self.tokenizer = tokenizer
        self.metadata = metadata
        # llama-server draws a random media marker at startup (`/props`), so text inside a
        # request can never stand for an image; Rizzo's own markers are swapped in when sent.
        self.marker = marker

    @classmethod
    def load(
        cls,
        model,
        projector,
        device="auto",
        ctx=8192,
        threads=None,
        runtime_dir=None,
        log_path=LOG_FILE,
    ):
        model, projector = Path(model).resolve(), Path(projector).resolve()
        for path in (model, projector):
            if not path.is_file():
                raise ValueError(
                    f"GGUF file not found at {path}. Run `rizzo download --vision` first."
                )
        if ctx < 1:
            raise ValueError("ctx must be positive")
        started = time.perf_counter()
        hashes = {path.name: llama_release.sha256_file(path) for path in (model, projector)}
        pin = next(
            (
                spec
                for spec in VISION.values()
                if hashes.get(spec.model.file) == spec.model.sha256
                and hashes.get(spec.projector.file) == spec.projector.sha256
            ),
            None,
        )
        family = None if device in ("auto", "gpu", "cpu") else device
        directory = Path(runtime_dir or llama_release.locate(family))
        binary = directory / (SERVER + (".exe" if os.name == "nt" else ""))
        if not binary.is_file():
            raise ValueError(f"{binary.name} not found in {directory}; reinstall the runtime")
        server = LlamaServer.start(
            lambda port: LlamaServer.command(
                binary, model, projector, port, ctx + IMAGE_CELLS, device, threads
            ),
            log_path,
        )
        try:
            props = server.client.get("/props")
            if not (props.get("modalities") or {}).get("vision"):
                raise ValueError(f"{projector.name} gives {model.name} no image input")
            template = props.get("chat_template")
            if not template:
                raise ValueError("The GGUF file carries no chat template")
            tokenizer = ServerTokenizer(server.client, template)
            marker = props.get("media_marker") or MEDIA_MARKER
        except Exception:
            server.close()
            raise
        identity = {
            "source": pin.source if pin else model.stem,
            "gguf_source": pin.model.repo if pin else None,  # None: not one of the pinned files
            "gguf_revision": pin.model.revision if pin else None,
            "source_files": hashes,
            "precision": pin.model.quant if pin else "unpinned",
            "device": "cpu" if device == "cpu" else "gpu",
            "backend": directory.name.rsplit("-", 1)[-1],  # llama-b11081-darwin-arm64-metal
            "runtime": SERVER,
            "llama_cpp_release": llama_release.RELEASE,
            "llama_cpp_commit": llama_release.COMMIT,
            "prompt_version": PROMPT_VERSION,
            "media_marker": MEDIA_MARKER,
        }
        settings = props.get("default_generation_settings") or {}
        metadata = {
            **identity,
            "fingerprint": _digest(identity),
            "device_name": None,
            "context_cells": settings.get("n_ctx") or ctx + IMAGE_CELLS,
            "load_seconds": time.perf_counter() - started,
        }
        return cls(server, tokenizer, metadata, marker)

    def score(self, prefix: list[int], jobs: list[Compiled], mode="shared", images=()):
        """One full pass per question: images change positions, so no prefix is shared."""
        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        started = time.perf_counter()
        result, counted = {}, {}
        prompt_seconds = 0.0
        below = 0
        for job in jobs:
            markers = job.prompt.count(MEDIA_MARKER)
            if markers != len(images):
                raise ValueError(
                    f"Question {job.id}: {len(images)} images, {markers} media markers in the prompt"
                )
            prompt = (
                {
                    "prompt_string": job.prompt.replace(MEDIA_MARKER, self.marker),
                    "multimodal_data": list(images),
                }
                if images
                else job.prompt
            )
            reply = self.server.client.post(
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": 1,
                    "n_probs": N_PROBS,
                    "post_sampling_probs": False,
                    "temperature": -1,
                    "cache_prompt": False,
                    "response_fields": RESPONSE_FIELDS,
                },
            )
            if reply.get("truncated"):
                raise ValueError(
                    f"Question {job.id}: the prompt exceeds the context; no truncation"
                )
            top = {
                entry["id"]: entry["logprob"]
                for entry in reply["completion_probabilities"][0]["top_logprobs"]
            }
            lowest = min(top.values())
            result[job.id] = [top.get(slot, lowest) for slot in job.slots]
            below += sum(slot not in top for slot in job.slots)
            counted[job.id] = int(reply.get("tokens_evaluated") or len(job.tokens))
            prompt_seconds += (reply.get("timings") or {}).get("prompt_ms", 0.0) / 1000
        timing = {
            "inference_seconds": time.perf_counter() - started,
            "prefill_seconds": prompt_seconds,
            "shared_prefix_tokens": 0,
            "evaluated_tokens_including_padding": sum(counted.values()),
            "logical_input_tokens": sum(counted.values()),
            "batches": len(jobs),
            "generated_tokens": 0,
            "images": len(images),
            "letters_below_top_n": below,
            "input_tokens": counted,  # per question; Engine moves it into each answer
        }
        return result, timing

    def close(self):
        self.server.close()


def _digest(identity: dict) -> str:
    return hashlib.sha256(canonical(identity).encode()).hexdigest()
