import base64
import math
import sys

import pytest
from fastapi.testclient import TestClient
from test_service import FakeBackend

from rizzo_flow import loader
from rizzo_flow.api import create_app
from rizzo_flow.backend_vision import LlamaServer, ServerTokenizer, VisionBackend
from rizzo_flow.engine import Engine
from rizzo_flow.prompts import MEDIA_MARKER, render_state

SERVER_MARKER = "<__media_Xy7__>"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(32)).decode()
TEMPLATE = (
    "{%- for m in messages %}{{ '<' + m.role + '>' + m.content }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<assistant>' }}{%- endif %}"
)


class FakeClient:
    """Stands in for llama-server: characters are tokens, letter B is the favourite."""

    def __init__(self, logprobs=None):
        self.calls = []
        self.logprobs = logprobs or {ord("A"): -2.0, ord("B"): -0.5, ord("x"): -1.0}

    def get(self, path):
        self.calls.append((path, None))
        return {
            "modalities": {"vision": True},
            "chat_template": TEMPLATE,
            "media_marker": SERVER_MARKER,
        }

    def post(self, path, body):
        self.calls.append((path, body))
        if path == "/tokenize":
            return {"tokens": [ord(c) for c in body["content"]]}
        top = [{"id": key, "logprob": value} for key, value in self.logprobs.items()]
        return {
            "completion_probabilities": [{"top_logprobs": top}],
            "tokens_evaluated": 1234,
            "timings": {"prompt_ms": 250.0},
            "truncated": False,
        }


class FakeServer:
    def __init__(self, client):
        self.client = client
        self.closed = False

    def close(self):
        self.closed = True


def vision_backend(client=None):
    client = client or FakeClient()
    metadata = {
        "fingerprint": "test-only",
        "source": "Qwen/Qwen3-VL-4B-Instruct",
        "precision": "q8_0",
    }
    backend = VisionBackend(
        FakeServer(client), ServerTokenizer(client, TEMPLATE), metadata, SERVER_MARKER
    )
    return backend, client


def choice(images, options=("left", "right", "stop")):
    return {
        "state": "Obstacle ahead on the left.",
        "images": images,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Which way?",
                "options": [{"id": key, "description": key} for key in options],
                "policy": {"allow_abstain": False},
            }
        },
    }


def test_images_go_to_llama_server_inside_the_evidence():
    backend, client = vision_backend()
    answer = Engine(backend).decide(choice([PNG]))["answers"]["move"]
    completion = next(body for path, body in client.calls if path == "/completion")
    prompt = completion["prompt"]
    assert prompt["multimodal_data"] == [PNG]
    assert MEDIA_MARKER not in prompt["prompt_string"]  # the server's own random marker instead
    assert f"<evidence>\n{SERVER_MARKER}\nObstacle ahead" in prompt["prompt_string"]
    assert completion["n_predict"] == 1 and completion["post_sampling_probs"] is False
    assert completion["cache_prompt"] is False
    # log-probabilities are logits minus one constant: the softmax over letters is unchanged
    weights = {"left": math.exp(-2.0), "right": math.exp(-0.5), "stop": math.exp(-2.0)}
    total = sum(weights.values())
    for key, weight in weights.items():
        assert answer["probabilities"][key] == pytest.approx(weight / total)
    assert answer["choice"] == "right"
    assert answer["input_tokens"] == 1234  # image tokens included, counted by the runtime


def test_a_letter_below_the_reported_top_gets_the_lowest_reported_logprob():
    backend, _ = vision_backend(FakeClient({ord("B"): -0.1, ord("A"): -3.0}))
    response = Engine(backend).decide(choice([PNG]))
    assert response["answers"]["move"]["option_logits"]["stop"] == pytest.approx(-3.0)
    assert response["timing"]["letters_below_top_n"] == 1
    assert "input_tokens" not in response["timing"]


def test_text_only_requests_still_work_on_the_vision_backend():
    backend, client = vision_backend()
    Engine(backend).decide(choice([]))
    completion = next(body for path, body in client.calls if path == "/completion")
    assert isinstance(completion["prompt"], str) and MEDIA_MARKER not in completion["prompt"]


def test_text_backends_refuse_images():
    with pytest.raises(ValueError, match="text only"):
        Engine(FakeBackend()).decide(choice([PNG]))


def test_systemone_carries_images_and_validates_them():
    backend, _ = vision_backend()
    body = {
        "state": "Obstacle ahead on the left.",
        "images": [PNG],
        "model": "rizzo-latest",
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Which way?",
                "criteria": {"L": None, "R": None},
            }
        },
    }
    with TestClient(create_app(Engine(backend))) as http:
        result = http.post("/v1/systemone", json=body)
        assert result.status_code == 200
        assert result.json()["model"] == "rizzo-qwen3-vl-4b-instruct-q8_0"
        assert result.json()["usage"]["input_tokens"] == 1234
        body["images"] = ["data:image/png;base64," + PNG]
        assert http.post("/v1/systemone", json=body).status_code == 422
        body["images"] = [base64.b64encode(b"GIF89a" + bytes(32)).decode()]
        assert http.post("/v1/systemone", json=body).status_code == 422


def test_media_markers_open_the_evidence_and_never_come_from_the_state():
    assert render_state("S", 2) == f"<evidence>\n{MEDIA_MARKER}\n{MEDIA_MARKER}\nS\n</evidence>"
    assert render_state("S") == "<evidence>\nS\n</evidence>"
    with pytest.raises(ValueError, match="media marker"):
        render_state(f"look {MEDIA_MARKER}", 1)


def test_server_command_line():
    command = LlamaServer.command("bin/llama-server", "m.gguf", "p.gguf", 9001, 10240)
    assert command[:5] == ["bin/llama-server", "--model", "m.gguf", "--mmproj", "p.gguf"]
    assert command[command.index("--port") + 1] == "9001"
    assert command[command.index("--ctx-size") + 1] == "10240"
    assert command[command.index("--n-gpu-layers") + 1] == "999"
    assert "--no-mmproj-offload" not in command
    on_cpu = LlamaServer.command("s", "m", "p", 1, 2, device="cpu", threads=6)
    assert on_cpu[on_cpu.index("--n-gpu-layers") + 1] == "0" and "--no-mmproj-offload" in on_cpu
    assert on_cpu[-2:] == ["--threads", "6"]


def test_a_server_that_dies_while_loading_is_reported(tmp_path):
    log = tmp_path / "llama-server.log"
    with pytest.raises(ValueError, match="exited with code 3"):
        LlamaServer.start(
            lambda port: [sys.executable, "-c", "raise SystemExit(3)"], log, timeout=10
        )


def test_loader_checks_vision_options(tmp_path):
    with pytest.raises(ValueError, match="--mmproj needs --model"):
        loader.load_backend(mmproj=tmp_path / "p.gguf")
    with pytest.raises(ValueError, match="llama.cpp only"):
        loader.load_backend("mlx", vision="qwen3-vl-4b")
    with pytest.raises(ValueError, match="--quant"):
        loader.load_backend(vision="qwen3-vl-4b", quant="q8_0")
    with pytest.raises(ValueError, match="rizzo download --vision"):
        loader.load_backend(model=tmp_path / "m.gguf", mmproj=tmp_path / "p.gguf")


def test_close_stops_the_child():
    backend, _ = vision_backend()
    backend.close()
    assert backend.server.closed
