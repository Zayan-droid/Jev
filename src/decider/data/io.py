from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from decider.schema import Record


def write_records(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # mtime=0 keeps the gzip header deterministic, so identical data gives identical hashes.
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for record in records:
            line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            gz.write(line.encode("utf-8") + b"\n")
            n += 1
    return n


def iter_records(path: Path) -> Iterator[Record]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.model_validate_json(line)


def read_records(path: Path) -> list[Record]:
    return list(iter_records(path))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_path(data_dir: Path, split: str) -> Path:
    return data_dir / f"{split}.jsonl.gz"


def load_manifest(data_dir: Path) -> dict:
    manifest: dict = json.loads((data_dir / "manifest.json").read_text())
    return manifest
