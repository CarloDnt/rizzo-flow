from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    size: str
    repo: str
    revision: str
    hidden_size: int  # identifies a checkpoint directory regardless of its path

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1]


# Same Spark2.5 architecture, tokenizer and 1M-token context; pinned original weights.
MODELS = {
    spec.size: spec
    for spec in (
        ModelSpec("4b", "XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4", 2560),
        ModelSpec(
            "1.7b", "XHToken/Spark-X2.5-1.7B", "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8", 2048
        ),
    )
}
DEFAULT_SIZE = "4b"
MODEL_ID = MODELS[DEFAULT_SIZE].repo
MODEL_REVISION = MODELS[DEFAULT_SIZE].revision
RUNTIME_REVISION = "de2b4379fa1e2f2e1f99d84c83f0e008f651d86c"


@dataclass(frozen=True)
class GgufSpec:
    """One file of the GGUF conversions published by the model's authors, pinned by hash."""

    size: str
    quant: str
    repo: str
    revision: str
    file: str
    sha256: str

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1] / self.file

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


_GGUF_4B = ("XHToken/Spark-X2.5-4B-GGUF", "9826e0be84e6e6e8b9668abc91421109a1df1e2d")
_GGUF_17B = ("XHToken/Spark-X2.5-1.7B-GGUF", "1f7fa33b1245c14730da39e125714ad3a327901b")
GGUF = {
    (spec.size, spec.quant): spec
    for spec in (
        GgufSpec(
            "4b",
            "q8_0",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q8_0.gguf",
            "5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2",
        ),
        GgufSpec(
            "4b",
            "q4_k_m",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q4_K_M.gguf",
            "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428",
        ),
        GgufSpec(
            "4b",
            "bf16",
            *_GGUF_4B,
            "Spark-X2.5-4B.gguf",
            "8cecf405a41a4a10f833530910c2e13fde9fb39c325c8afc3c5d10e4181e1a14",
        ),
        GgufSpec(
            "1.7b",
            "q8_0",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q8_0.gguf",
            "cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442",
        ),
        GgufSpec(
            "1.7b",
            "q4_k_m",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q4_K_M.gguf",
            "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4",
        ),
        GgufSpec(
            "1.7b",
            "bf16",
            *_GGUF_17B,
            "Spark-X2.5-1.7B.gguf",
            "67d5f2f06e6d898efcf0dc40cab8528bc82b871c8dafb0936784183d2c10cdd9",
        ),
    )
}
QUANTS = ("q8_0", "q4_k_m", "bf16")
DEFAULT_QUANT = "q8_0"
DEFAULT_MODEL_PATH = MODELS[DEFAULT_SIZE].path


@dataclass(frozen=True)
class VisionSpec:
    """A vision-language checkpoint: language model and image projector (mmproj) as GGUF files.

    Served through llama.cpp's own server (`backend_vision`), not the in-process ctypes path."""

    name: str
    source: str  # original checkpoint, reported as the model's source
    model: GgufSpec
    projector: GgufSpec


_GGUF_QWEN3_VL_4B = ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "1cd86afb9a95c410a6038ab3b40d8b578c892266")
VISION = {
    spec.name: spec
    for spec in (
        VisionSpec(
            "qwen3-vl-4b",
            "Qwen/Qwen3-VL-4B-Instruct",
            GgufSpec(
                "qwen3-vl-4b",
                "q8_0",
                *_GGUF_QWEN3_VL_4B,
                "Qwen3VL-4B-Instruct-Q8_0.gguf",
                "054721f478bc5fa6beffb7f38eae575d45298f88cbb8d2f83ef675a727863eb1",
            ),
            GgufSpec(
                "qwen3-vl-4b",
                "f16",
                *_GGUF_QWEN3_VL_4B,
                "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
                "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331",
            ),
        ),
    )
}
DEFAULT_VISION = "qwen3-vl-4b"


def identify(config: dict) -> ModelSpec:
    """Match a checkpoint's config.json to a supported, pinned model."""
    for spec in MODELS.values():
        if config.get("hidden_size") == spec.hidden_size:
            return spec
    raise ValueError(
        f"Unrecognized Spark2.5 checkpoint (hidden_size={config.get('hidden_size')}); "
        f"supported sizes: {', '.join(MODELS)}"
    )


def download_gguf(size=DEFAULT_SIZE, quant=DEFAULT_QUANT, destination=None, progress=None):
    """Fetch one pinned GGUF file, verified against its sha256."""
    from .llama_release import fetch

    spec = GGUF[(size, quant)]
    return fetch(spec.url, Path(destination) if destination else spec.path, spec.sha256, progress)


def download_vision(name=DEFAULT_VISION, progress=None):
    """Fetch the pinned model and projector of a vision checkpoint, each verified by sha256."""
    from .llama_release import fetch

    spec = VISION[name]
    return [
        fetch(part.url, part.path, part.sha256, progress) for part in (spec.model, spec.projector)
    ]


def download_model(destination=None, size=DEFAULT_SIZE):
    from huggingface_hub import snapshot_download

    spec = MODELS[size]
    return snapshot_download(
        spec.repo,
        revision=spec.revision,
        local_dir=destination or spec.path,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.safetensors",
            "tokenizer.model",
            "LICENSE",
            "README.md",
        ],
    )
