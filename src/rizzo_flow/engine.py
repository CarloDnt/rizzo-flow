"""Thread-safe long-lived model service with deterministic postprocessing."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .decisions import decode
from .prompts import compile_request
from .responses import Response
from .schema import Request


class Engine:
    def __init__(self, backend, ctx=8192, calibration=None):
        if ctx < 1:
            raise ValueError("ctx must be positive")
        self.backend = backend
        self.ctx = ctx
        self.calibration = calibration
        if calibration and calibration.fingerprint != backend.metadata["fingerprint"]:
            raise ValueError(
                "Calibration was fitted for a different model/runtime/prompt configuration"
            )
        self._lock = Lock()
        # All model work runs on one thread that lives as long as the process. Web servers call
        # decide() from short-lived pool threads, and MLX's CUDA backend aborts the process when
        # a thread that ran computations exits.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rizzo-inference")

    def decide(self, request: Request | dict) -> dict:
        # Re-validate a serialized snapshot, also protecting mutable Pydantic objects.
        request = Request.model_validate(
            request.model_dump() if isinstance(request, Request) else request
        )
        vision = getattr(self.backend, "vision", False)
        if request.images and not vision:
            raise ValueError(
                "This model reads text only; images need a vision model (`rizzo serve --vision`)"
            )
        # Only a vision backend takes images; the text backends keep their signature.
        extra = {"images": request.images} if vision else {}
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            prefix, jobs = compile_request(self.backend.tokenizer, request, self.ctx)
            encoded = time.perf_counter()
            logits, timing = self._worker.submit(
                self.backend.score, prefix, jobs, request.mode, **extra
            ).result()
            # Image tokens are counted by the runtime, not by the tokenizer.
            counted = timing.pop("input_tokens", {})
            answers = {}
            for job in jobs:
                question = request.questions[job.id]
                temperature = (
                    self.calibration.temperatures.get(question.type, 1.0)
                    if self.calibration
                    else 1.0
                )
                answers[job.id] = decode(question, logits[job.id], temperature)
                answers[job.id]["prompt_sha256"] = job.prompt_sha256
                answers[job.id]["input_tokens"] = counted.get(job.id, len(job.tokens))
        response = {
            "model": self.backend.metadata,
            "mode": request.mode,
            "answers": answers,
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "timing": {
                **timing,
                "queue_seconds": acquired - started,
                "compile_seconds": encoded - acquired,
                "total_seconds": time.perf_counter() - started,
            },
        }
        return Response.model_validate(response).model_dump()
