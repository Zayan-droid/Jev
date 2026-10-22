"""decider-label driven with scripted answers."""

import io
import json
from pathlib import Path

import pytest

from decider.cli import label
from decider.data.build import BuildConfig, build_dataset
from decider.schema import Record


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    cfg = BuildConfig.from_dict(
        {
            "name": "lbl",
            "output_root": str(tmp_path),
            "synthetic": {"families": ["inventory", "arithmetic"]},
            "synthetic_sizes": {"test": 30},
        }
    )
    return build_dataset(cfg)


def run(monkeypatch: pytest.MonkeyPatch, argv: list[str], answers: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(answers))
    label.main(argv)


def test_sampling_is_stratified_and_hides_labels(data_dir: Path) -> None:
    entries = label.sample_entries(data_dir, "test", n=6, seed=0)
    assert len(entries) == 6
    assert {e["family"] for e in entries} == {"inventory", "arithmetic"}
    assert all("answers" not in e and e["source"] == "gold" for e in entries)


def test_gold_labelling_writes_records_and_resumes(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "gold.jsonl"
    argv = ["--sample-from", str(data_dir), "-n", "2", "--split", "gold", "--out", str(out)]
    entries = label.sample_entries(data_dir, "test", n=2, seed=0)
    first = entries[0]
    # Answer every question of the first record with its first option ("y" for noul),
    # then quit.
    answers = "".join("y\n" if q["type"] == "noul" else "0\n" for q in first["questions"].values())
    run(monkeypatch, argv, answers + "q\n")
    records = [Record.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0].split == "gold" and records[0].label_source == "human:gold"
    assert records[0].family == first["family"]
    assert records[0].meta["request_id"] == first["request_id"]

    run(monkeypatch, argv, "q\n")  # resumes: the labelled record is not asked again
    assert len(out.read_text().splitlines()) == 1


def test_queue_labelling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = tmp_path / "queue.jsonl"
    entry = {
        "request_id": "abc",
        "model_version": "m1",
        "state": {"ticket": "broken"},
        "questions": {"refund": {"type": "noul", "instructions": "Refund?"}},
        "answers": {"refund": {"type": "noul", "p_yes": 0.5}},
        "review": ["refund"],
    }
    queue.write_text(json.dumps(entry) + "\n")
    out = tmp_path / "labeled.jsonl"
    run(monkeypatch, ["--queue", str(queue), "--out", str(out)], "n\n")
    record = Record.model_validate_json(out.read_text())
    assert record.targets == {"refund": 0.0} and record.source == "feedback"
