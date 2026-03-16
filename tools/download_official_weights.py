import argparse
from pathlib import Path


VMAMBA_PRETRAIN_ID = "1O9P6XLuWtUxFa70vwrYCRVedRAFutczV"
REMAMBER_MAMBA_ID = "1CqkBL5Dqm4X3ZgPFfO026l9liCAbeEiD"
REMAMBER_CONV_ID = "1QeS_iq1i4VaTE3nqbUVgsbNwgKllDQRQ"
CLIP_REPO_ID = "openai/clip-vit-large-patch14"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def google_drive_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"


def download_file(file_id, output_path):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required to download Google Drive assets.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Skipping existing file: {output_path}")
        return
    gdown.download(google_drive_url(file_id), str(output_path), quiet=False, fuzzy=True)


def prefetch_clip(cache_dir=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to prefetch CLIP weights.") from exc

    snapshot_download(
        repo_id=CLIP_REPO_ID,
        cache_dir=cache_dir,
        local_dir=None,
        local_dir_use_symlinks=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download official ReMamber assets for server setup.")
    parser.add_argument("--pretrain-dir", default="pretrain", help="Directory for the required VMamba pretrain checkpoint.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for optional official ReMamber checkpoints.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory for CLIP prefetch.")
    parser.add_argument(
        "--skip-task-checkpoints",
        action="store_true",
        help="Skip downloading the official ReMamber task checkpoints.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip prefetching the CLIP text encoder assets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pretrain_dir = ensure_dir(Path(args.pretrain_dir))
    checkpoint_dir = ensure_dir(Path(args.checkpoint_dir))

    download_file(
        VMAMBA_PRETRAIN_ID,
        pretrain_dir / "vssm_base_0229_ckpt_epoch_237.pth",
    )

    if not args.skip_task_checkpoints:
        download_file(REMAMBER_MAMBA_ID, checkpoint_dir / "ReMamber_Mamba.pth")
        download_file(REMAMBER_CONV_ID, checkpoint_dir / "ReMamber_Conv.pth")

    if not args.skip_clip:
        prefetch_clip(cache_dir=args.hf_cache_dir)


if __name__ == "__main__":
    main()
