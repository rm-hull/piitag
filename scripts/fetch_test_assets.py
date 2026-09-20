"""Download the pinned Redact model assets for local tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "desert-ant-labs/redact"


def _get_revision() -> str:
    """Read revision from .model-revision file."""
    # Try package location first (for installed packages)
    import piitag

    package_dir = Path(piitag.__file__).parent
    revision_file = package_dir / ".model-revision"
    if revision_file.exists():
        return revision_file.read_text().strip()

    # Fallback to development location
    revision_file = Path("src/piitag/.model-revision")
    if revision_file.exists():
        return revision_file.read_text().strip()

    # Final fallback
    return "v0.4.0"


REVISION = _get_revision()
ASSETS = (
    "redact_tokenizer.bin",
    "labels.json",
    "redact.tflite",
)


def fetch_assets(destination: Path) -> None:
    """Download model assets into ``destination`` without bundling them."""

    destination.mkdir(parents=True, exist_ok=True)
    for filename in ASSETS:
        cached_path = hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            revision=REVISION,
        )
        target = destination / filename
        target.write_bytes(Path(cached_path).read_bytes())
        print(f"Downloaded {filename} -> {target} ({REVISION})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/model"),
        help="directory for local model assets (default: tests/fixtures/model)",
    )
    args = parser.parse_args()
    fetch_assets(args.output)


if __name__ == "__main__":
    main()
