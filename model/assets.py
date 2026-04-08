from pathlib import Path


CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VMAMBA_BASE_CKPT = "vssm_base_0229_ckpt_epoch_237.pth"
VMAMBA_BASE_URL = "https://drive.google.com/file/d/1O9P6XLuWtUxFa70vwrYCRVedRAFutczV/view?usp=sharing"


def resolve_pretrain_path(pretrain_path):
    path = Path(pretrain_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_hf_cache_dir(pretrain_path, hf_cache_dir=""):
    if hf_cache_dir:
        cache_dir = Path(hf_cache_dir).expanduser()
    else:
        cache_dir = resolve_pretrain_path(pretrain_path) / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def ensure_vmamba_checkpoint(pretrain_path, model_size="base", download_url=""):
    if model_size != "base":
        raise ValueError(f"Unsupported model_size for automatic VMamba checkpoint provisioning: {model_size}")

    ckpt_path = resolve_pretrain_path(pretrain_path) / VMAMBA_BASE_CKPT
    if ckpt_path.exists():
        return ckpt_path

    url = download_url or VMAMBA_BASE_URL
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Automatic VMamba checkpoint download requires gdown. "
            "Install the environment artifact for this repository before training."
        ) from exc

    gdown.download(url=url, output=str(ckpt_path), quiet=False, fuzzy=True)
    if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
        raise RuntimeError(f"VMamba checkpoint download did not produce a valid file: {ckpt_path}")

    return ckpt_path
