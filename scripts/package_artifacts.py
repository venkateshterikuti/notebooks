#!/usr/bin/env python
"""Collect training artifacts, build a manifest, and optionally sync to local storage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="Directory to collect artifacts into")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint file(s) to include")
    parser.add_argument("--include", action="append", default=[], help="Additional files to include")
    parser.add_argument("--archive", type=str, default=None, help="Path of the .tar.gz archive to create")
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        help="Optional rsync/scp destination (for example user@host:/path)",
    )
    parser.add_argument(
        "--download-method",
        type=str,
        default="rsync",
        choices=["rsync", "scp"],
        help="Command used to push the archive to --download",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Name of the manifest file written inside the artifact directory",
    )
    return parser.parse_args()


def _iter_paths(values: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        if not value:
            continue
        paths.append(Path(value).expanduser())
    return paths


def _stage_path(src: Path, dest_dir: Path) -> Tuple[Path, Path] | None:
    if not src.exists():
        print(f"[warn] skipping missing artifact: {src}")
        return None
    src = src.resolve()
    dest_dir = dest_dir.resolve()
    if dest_dir == src or dest_dir in src.parents:
        return src, src
    dest = dest_dir / src.name
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest, src


def _walk_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _human_bytes(num: int) -> str:
    step = 1024.0
    size = float(num)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < step or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= step
    return f"{size:.2f} PB"


def main() -> None:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    staged: Dict[Path, Dict[str, Path]] = {}
    for candidate in _iter_paths(args.checkpoint) + _iter_paths(args.include):
        result = _stage_path(candidate, artifact_dir)
        if result is None:
            continue
        staged_path, original_path = result
        key = staged_path.resolve()
        if key in staged:
            continue
        staged[key] = {"dest": staged_path, "source": original_path}

    manifest_entries = []
    for entry in staged.values():
        dest = entry["dest"].resolve()
        source = entry["source"].resolve()
        try:
            relative = dest.relative_to(artifact_dir)
        except ValueError:
            relative = dest.name
        size_bytes = _walk_size(dest)
        manifest_entries.append(
            {
                "name": dest.name,
                "relative_path": str(relative),
                "size_bytes": size_bytes,
                "size_human": _human_bytes(size_bytes),
                "source": str(source),
            }
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "artifact_dir": str(artifact_dir),
        "entries": manifest_entries,
    }

    manifest_path = artifact_dir / args.manifest_name
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[manifest] wrote {manifest_path}")

    archive_path = Path(args.archive) if args.archive else artifact_dir.with_suffix(".tar.gz")
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(artifact_dir, arcname=artifact_dir.name)
    print(f"[archive] created {archive_path}")

    manifest["archive"] = str(archive_path)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    if args.download:
        cmd = (
            ["rsync", "-av", str(archive_path), args.download]
            if args.download_method == "rsync"
            else ["scp", str(archive_path), args.download]
        )
        print(f"[download] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                "[download] WARNING: copy failed with exit code"
                f" {result.returncode}. Transfer the archive manually."
            )
        else:
            print(f"[download] archive copied to {args.download}")


if __name__ == "__main__":
    main()
