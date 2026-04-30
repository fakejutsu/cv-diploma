from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models.original_wavevit_backbone import _OFFICIAL_PRETRAINED_ALIASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official WaveViT ImageNet-1K pretrained checkpoints.")
    parser.add_argument(
        "--alias",
        default="imagenet_1k_224",
        choices=tuple(sorted(_OFFICIAL_PRETRAINED_ALIASES)),
        help="Pretrained checkpoint alias.",
    )
    parser.add_argument(
        "--variant",
        default="wavevit_s",
        choices=("wavevit_s", "wavevit_b", "wavevit_l"),
        help="WaveViT model variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights/pretrained/wavevit"),
        help="Directory where the checkpoint will be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = _OFFICIAL_PRETRAINED_ALIASES[args.alias][args.variant]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(entry["filename"])
    if output_path.is_file():
        print(f"Checkpoint already exists: {output_path}")
        return 0

    try:
        import gdown
    except ImportError:
        print(
            "gdown is required to download Google Drive checkpoints automatically.\n"
            "Install it with `pip install gdown`, or download manually:\n"
            f"  Google Drive: {entry['google_drive_url']}\n"
            f"  Baidu: {entry['baidu_url']} (access code: nets)\n"
            f"Then save it as: {output_path}",
            file=sys.stderr,
        )
        return 2

    print(f"Downloading {args.variant} {args.alias} to {output_path}")
    gdown.download(id=str(entry["google_drive_id"]), output=str(output_path), quiet=False)
    if not output_path.is_file():
        print(f"Download did not create expected file: {output_path}", file=sys.stderr)
        return 1
    print(f"Saved checkpoint: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
