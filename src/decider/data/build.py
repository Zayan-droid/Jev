"""Build a versioned dataset: public sources + programmatic tasks -> clean, split JSONL.

Output (``<output_root>/<name>/``)::

    train.jsonl.gz  val.jsonl.gz  calib.jsonl.gz  test.jsonl.gz  heldout.jsonl.gz
    manifest.json   counts, per-file SHA-256, quality report, config and code version

Split roles:

* ``val``     model selection and early stopping
* ``calib``   fitting temperatures (kept apart from ``val`` so calibration is not tuned on the
  data used to pick the checkpoint)
* ``test``    final numbers on seen task families
* ``heldout`` unseen datasets, families and question templates (zero-shot)
* ``gold``    hand-labelled records (``local_sources``), never trained on
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from decider.data.converters import CONVERTERS
from decider.data.io import sha256_file, split_path, write_records
from decider.data.quality import assert_no_leakage, deduplicate, summarize
from decider.data.synthetic import GEN_VERSION, SyntheticConfig, SyntheticGenerator
from decider.schema import Record
from decider.utils.provenance import git_info

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "calib", "test", "heldout", "gold")


@dataclass
class SourceConfig:
    name: str
    max_train: int | None = None  # rows drawn from the source's training split
    max_eval: int | None = None  # rows drawn from each evaluation split


@dataclass
class LocalSource:
    """A JSONL file of canonical records, e.g. a hand-labelled gold set or reviewed feedback.
    Missing files are skipped, so configs can reference sets that do not exist yet."""

    path: str
    split: str = "train"


@dataclass
class SyntheticSplitSizes:
    train: int = 0
    val: int = 0
    calib: int = 0
    test: int = 0
    heldout: int = 0


@dataclass
class BuildConfig:
    name: str
    output_root: str = "data/processed"
    seed: int = 0
    val_fraction: float = 0.03
    calib_fraction: float = 0.03
    sources: list[SourceConfig] = field(default_factory=list)
    local_sources: list[LocalSource] = field(default_factory=list)
    synthetic: SyntheticConfig | None = None
    synthetic_sizes: SyntheticSplitSizes = field(default_factory=SyntheticSplitSizes)
    hf_cache_dir: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildConfig:
        data = dict(data)
        sources = [SourceConfig(**s) for s in data.pop("sources", []) or []]
        local = [LocalSource(**s) for s in data.pop("local_sources", []) or []]
        synthetic = data.pop("synthetic", None)
        sizes = SyntheticSplitSizes(**(data.pop("synthetic_sizes", None) or {}))
        synth_cfg = None
        if synthetic is not None:
            fields: dict[str, Any] = {
                k: tuple(v) if isinstance(v, list) else v for k, v in synthetic.items()
            }
            synth_cfg = SyntheticConfig(**fields)
        return cls(
            sources=sources,
            local_sources=local,
            synthetic=synth_cfg,
            synthetic_sizes=sizes,
            **data,
        )


def _bucket(rid: str) -> float:
    return int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _train_split(rid: str, cfg: BuildConfig) -> str:
    u = _bucket(rid)
    if u < cfg.val_fraction:
        return "val"
    if u < cfg.val_fraction + cfg.calib_fraction:
        return "calib"
    return "train"


def convert_source(source: SourceConfig, cfg: BuildConfig) -> list[Record]:
    from datasets import load_dataset

    converter = CONVERTERS[source.name]()
    records: list[Record] = []
    for hf_split, role in converter.splits.items():
        ds = load_dataset(
            converter.hf_path, converter.hf_config, split=hf_split, cache_dir=cfg.hf_cache_dir
        )
        converter.setup(ds)
        cap = source.max_train if role == "train" else source.max_eval
        ds = ds.shuffle(seed=cfg.seed)
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        for i, row in enumerate(ds):
            rid = f"{source.name}-{hf_split}-{i:07d}"
            split = _train_split(rid, cfg) if role == "train" else role
            record = converter.convert(row, rid, split)
            if record is not None:
                records.append(record)
        log.info("%s/%s: %d rows -> %s", source.name, hf_split, len(ds), role)
    return records


def build_dataset(cfg: BuildConfig) -> Path:
    out_dir = Path(cfg.output_root) / cfg.name
    by_split: dict[str, list[Record]] = {s: [] for s in SPLITS}
    licenses: dict[str, str] = {}

    for source in cfg.sources:
        licenses[source.name] = CONVERTERS[source.name].license
        for record in convert_source(source, cfg):
            by_split[record.split].append(record)

    for local in cfg.local_sources:
        path = Path(local.path)
        if not path.exists():
            log.warning("local source %s does not exist; skipping", path)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = Record.model_validate_json(line)
                by_split[local.split].append(record.model_copy(update={"split": local.split}))
        licenses[str(path)] = "local"

    if cfg.synthetic is not None:
        generator = SyntheticGenerator(cfg.synthetic)
        for split in SPLITS:
            n = getattr(cfg.synthetic_sizes, split, 0)
            by_split[split].extend(generator.generate(n, split, cfg.seed))
        licenses["synthetic"] = "generated (MIT, this repository)"

    by_split, report = deduplicate(by_split)
    assert_no_leakage(by_split)

    files: dict[str, dict[str, Any]] = {}
    for split, records in by_split.items():
        path = split_path(out_dir, split)
        write_records(path, records)
        files[split] = {"path": path.name, "sha256": sha256_file(path), **summarize(records)}
        log.info("%s: %d records", split, len(records))

    content_hash = hashlib.sha256("".join(f["sha256"] for f in files.values()).encode())
    manifest = {
        "name": cfg.name,
        "data_hash": content_hash.hexdigest()[:16],
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "generator_version": GEN_VERSION,
        "code": git_info(),
        "config": asdict(cfg),
        "licenses": licenses,
        "quality": report.to_dict(),
        "splits": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir
