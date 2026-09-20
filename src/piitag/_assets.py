"""Resolve the pinned Redact model assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "desert-ant-labs/redact"


def _get_revision() -> str:
    """Read revision from .model-revision file."""
    # Look for .model-revision in the package directory
    revision_file = Path(__file__).parent / ".model-revision"
    if revision_file.exists():
        return revision_file.read_text().strip()
    # Fallback to default
    return "v0.4.0"


REVISION = _get_revision()


@dataclass(frozen=True)
class Assets:
    """Paths to the files required by the Redact model."""

    tokenizer: Path
    labels: Path
    tflite: Path

    @property
    def directory(self) -> Path:
        """Return the common directory when all assets share one."""
        return self.tokenizer.parent


ASSET_FILENAMES = {
    "tokenizer": "redact_tokenizer.bin",
    "labels": "labels.json",
    "tflite": "redact.tflite",
}


def _from_directory(directory: Path) -> Assets:
    paths = {name: directory / filename for name, filename in ASSET_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Redact model assets are missing from the supplied directory: "
            + ", ".join(missing)
        )
    return Assets(**paths)


def resolve_assets(directory: str | Path | None = None) -> Assets:
    """Resolve local assets or download the pinned files into the HF cache.

    A supplied directory is strictly offline: it must already contain all
    three files. Without a directory, ``huggingface_hub`` manages the cache.
    """
    if directory is not None:
        return _from_directory(Path(directory))
    downloaded = {
        name: Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                revision=REVISION,
            )
        )
        for name, filename in ASSET_FILENAMES.items()
    }
    return Assets(**downloaded)


__all__ = ["ASSET_FILENAMES", "Assets", "resolve_assets"]
