"""``decider-label``: label requests in the terminal.

Two sources of work:

* the serving review queue (default): low-confidence production requests, shown with the
  model's answer; labels go to ``data/feedback/labeled.jsonl`` and into the next training set;
* ``--sample-from <dataset dir>``: a family-stratified sample of an existing split, shown
  without any label or model answer so nothing anchors the judgement; with ``--split gold``
  this builds the hand-labelled evaluation set ``data/gold/gold.jsonl``.

Both outputs are canonical records listed under ``local_sources`` in the dataset configs.
Labelling can stop and resume at any time; finished requests are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from decider.cli._common import configure_logging
from decider.schema import Record

DEFAULT_OUT = {"train": Path("data/feedback/labeled.jsonl"), "gold": Path("data/gold/gold.jsonl")}


def _ask_target(console: Any, question: dict[str, Any]) -> Any | None:
    """Returns a target, ``None`` to skip, or raises ``KeyboardInterrupt`` to stop."""
    from rich.prompt import Prompt

    if question["type"] == "noul":
        answer = Prompt.ask("  answer", choices=["y", "n", "s", "q"], console=console)
        if answer == "q":
            raise KeyboardInterrupt
        return None if answer == "s" else float(answer == "y")
    names = list(question["criteria"]) if question["type"] == "choice" else question["levels"]
    for i, name in enumerate(names):
        console.print(f"    [{i}] {name}")
    answer = Prompt.ask("  index (s=skip, q=quit)", console=console)
    if answer == "q":
        raise KeyboardInterrupt
    if answer == "s" or not answer.isdigit() or int(answer) >= len(names):
        return None
    k = int(answer)
    if question["type"] == "choice":
        return {names[k]: 1.0}
    return [float(i == k) for i in range(len(names))]


def queue_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries.append({**entry, "source": "feedback", "family": "feedback"})
    return entries


def sample_entries(data_dir: Path, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Up to ``n`` records, round-robin across families so every family is represented."""
    from decider.data.io import iter_records, split_path

    by_family: dict[str, list[Record]] = defaultdict(list)
    for record in iter_records(split_path(data_dir, split)):
        by_family[record.family].append(record)
    rng = random.Random(seed)
    for records in by_family.values():
        rng.shuffle(records)
    picked: list[Record] = []
    while len(picked) < n and any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family] and len(picked) < n:
                picked.append(by_family[family].pop())
    return [
        {
            "request_id": r.id,
            "model_version": None,
            "source": "gold",
            "family": r.family,
            "state": r.state,
            "questions": {k: q.model_dump() for k, q in r.questions.items()},
            "review": list(r.questions),
        }
        for r in picked
    ]


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--queue", type=Path, default=Path("data/feedback/queue.jsonl"))
    p.add_argument("--sample-from", type=Path, help="dataset directory to sample records from")
    p.add_argument("--sample-split", default="test")
    p.add_argument("-n", type=int, default=50, help="records to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="train", choices=["train", "gold"])
    p.add_argument("--out", type=Path, help="default: by --split, see the description")
    args = p.parse_args(argv)
    out_path: Path = args.out or DEFAULT_OUT[args.split]

    from rich.console import Console
    from rich.panel import Panel
    from rich.pretty import Pretty

    console = Console()
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["meta"].get("request_id", ""))
    if args.sample_from:
        entries = sample_entries(args.sample_from, args.sample_split, args.n, args.seed)
    else:
        entries = queue_entries(args.queue)
    todo = [e for e in entries if e["request_id"] not in done]
    console.print(f"{len(todo)} requests to label ({len(done)} already in {out_path})")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labelled = 0
    with out_path.open("a", encoding="utf-8") as out:
        try:
            for entry in todo:
                console.print(Panel(Pretty(entry["state"]), title=entry["request_id"][:24]))
                targets: dict[str, Any] = {}
                for name in entry["review"]:
                    question = entry["questions"][name]
                    console.print(
                        f"[bold]{name}[/] ({question['type']}): {question['instructions']}"
                    )
                    if entry.get("answers"):
                        console.print(f"  model: {entry['answers'][name]}")
                    target = _ask_target(console, question)
                    if target is not None:
                        targets[name] = target
                if not targets:
                    continue
                record = Record(
                    id=f"{entry['source']}-{entry['request_id']}",
                    source=entry["source"],
                    family=entry["family"],
                    split=args.split,
                    state=entry["state"],
                    questions={k: entry["questions"][k] for k in targets},
                    targets=targets,
                    label_source="human:review" if entry["source"] == "feedback" else "human:gold",
                    meta={"request_id": entry["request_id"], "model": entry["model_version"]},
                )
                out.write(record.model_dump_json() + "\n")
                out.flush()
                labelled += 1
        except KeyboardInterrupt:
            console.print("\nstopped")
    console.print(f"labelled {labelled} requests -> {out_path}")


if __name__ == "__main__":
    main()
