"""Check for model revision updates and output current/latest versions."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi


def get_current_revision(filepath: Path) -> str:
    """Read the current revision from the .model-revision file."""
    if filepath.exists():
        return filepath.read_text().strip()
    # Fallback to default
    return "v0.4.0"


def get_latest_revision(repo_id: str) -> str:
    """Get the latest version tag from HuggingFace Hub."""
    api = HfApi()

    # Get repository refs (includes tags)
    refs = api.list_repo_refs(repo_id)

    # Get tags that start with 'v'
    tags = [tag.name for tag in refs.tags if tag.name.startswith("v")]

    if not tags:
        raise ValueError(f"No version tags found in {repo_id}")

    def version_key(tag: str) -> tuple[int, int, int]:
        match = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if match:
            return tuple(int(x) for x in match.groups())
        return (0, 0, 0)

    return max(tags, key=version_key)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string to tuple for comparison."""
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return tuple(int(x) for x in match.groups())
    return (0, 0, 0)


def main() -> int:
    # Get environment variables
    # Default to the installed package location, or fall back to root
    revision_file = Path(os.environ.get("REVISION_FILE", "src/piitag/.model-revision"))
    model_repo = os.environ.get("MODEL_REPO", "desert-ant-labs/redact")

    # Get versions
    current = get_current_revision(revision_file)
    latest = get_latest_revision(model_repo)

    print(f"current={current}")
    print(f"latest={latest}")
    print(f"version_file={revision_file}")
    print(f"model_repo={model_repo}")

    # Compare versions
    if current == latest:
        print("update_needed=false")
        return 0

    curr_tuple = parse_version(current)
    latest_tuple = parse_version(latest)

    if curr_tuple >= latest_tuple:
        print("update_needed=false")
        return 0

    print("update_needed=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
