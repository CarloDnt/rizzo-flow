"""Choose and load the scoring backend: llama.cpp by default, MLX on request."""

from pathlib import Path

from .config import DEFAULT_QUANT, DEFAULT_SIZE, DEFAULT_VISION, GGUF, MODELS, VISION

BACKENDS = ("llama", "mlx")
# `auto`, `gpu` and `cpu` work everywhere. The other names ask for one GPU family: `mlx` and
# `cuda` also keep their meaning for the MLX backend, the rest exist in llama.cpp only.
DEVICES = ("auto", "gpu", "cpu", "cuda", "metal", "vulkan", "rocm", "sycl", "mlx")
MLX_DEVICES = ("auto", "gpu", "cpu", "cuda", "mlx")


def load_backend(
    backend="llama",
    *,
    size=DEFAULT_SIZE,
    model=None,
    quant=None,
    bits=None,
    device="auto",
    ctx=8192,
    batch_size=4,
    threads=None,
    vision=None,
    mmproj=None,
):
    if backend not in BACKENDS:
        raise ValueError(f"Backend must be one of: {', '.join(BACKENDS)}")
    if vision or mmproj:
        return _load_vision(backend, vision, model, mmproj, quant, bits, device, ctx, threads)
    if backend == "mlx":
        if quant:
            raise ValueError("--quant selects a GGUF file (llama backend); with MLX use --bits 4|8")
        if device not in MLX_DEVICES:
            raise ValueError(f"--device {device} exists only in the llama backend")
        from .backend import SparkBackend

        return SparkBackend.load(
            model or MODELS[size].path, bits=bits, device=device, batch_size=batch_size
        )
    if bits:
        raise ValueError(
            "--bits quantizes MLX weights in memory (--backend mlx); llama.cpp loads a "
            "quantized file instead: --quant q8_0 (default), q4_k_m or bf16"
        )
    if device == "mlx":
        raise ValueError("--device mlx needs --backend mlx; with llama.cpp use --device metal")
    if model and Path(model).is_dir():
        raise ValueError(
            f"{model} is a checkpoint directory (MLX); llama.cpp needs a .gguf file. "
            "Pass --backend mlx, or a GGUF path, or drop --model to use the pinned file."
        )
    from .backend_llama import LlamaBackend

    return LlamaBackend.load(
        model or GGUF[(size, quant or DEFAULT_QUANT)].path,
        device=device,
        ctx=ctx,
        batch_size=batch_size,
        threads=threads,
    )


def _load_vision(backend, vision, model, mmproj, quant, bits, device, ctx, threads):
    """A vision-language model: the pinned pair (`--vision`) or `--model` with `--mmproj`."""
    if backend != "llama":
        raise ValueError("Vision models run on llama.cpp only: drop --backend mlx")
    if quant or bits:
        raise ValueError(
            "--quant and --bits pick Spark files; a vision model is --vision or --mmproj"
        )
    if device == "mlx":
        raise ValueError("--device mlx needs --backend mlx; with llama.cpp use --device metal")
    if mmproj and not model:
        raise ValueError("--mmproj needs --model, the language model GGUF it belongs to")
    if vision and vision not in VISION:
        raise ValueError(f"Vision models: {', '.join(VISION)}")
    spec = VISION[vision or DEFAULT_VISION]
    from .backend_vision import VisionBackend

    return VisionBackend.load(
        model or spec.model.path,
        mmproj or spec.projector.path,
        device=device,
        ctx=ctx,
        threads=threads,
    )


def describe() -> dict:
    """What `rizzo devices` prints: the llama.cpp runtime and its devices, and MLX if present."""
    from . import llama_release
    from .llama_cpp import Library, choose_device

    report = {
        "llama.cpp": {
            "release": llama_release.RELEASE,
            "host": "/".join(llama_release.host()),
            "packages": llama_release.supported(),
            "recommended": None,
            "installed": llama_release.installed(),
        }
    }
    section = report["llama.cpp"]
    try:
        section["recommended"] = llama_release.pick("auto")
        section["directory"] = str(llama_release.locate())
        devices = Library.open().devices()
        chosen = choose_device(devices, "auto")
        section["devices"] = [device.public() for device in devices]
        section["auto_selects"] = chosen.name if chosen else "CPU"
    except (ValueError, OSError) as error:
        section["error"] = str(error)
    try:
        from .runtime import describe as describe_mlx

        report["mlx"] = describe_mlx()
    except ImportError:
        report["mlx"] = {"installed": False}
    return report
