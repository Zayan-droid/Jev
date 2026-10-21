"""``decider-label``: label queued low-confidence predictions in the terminal.

Reads the serving feedback queue (or any file in the same format), shows each state with the
question and the model's answer, and writes labelled records in the canonical schema. Point a
dataset's ``local_sources`` at the output to fold the labels into the next data version, or
write to a ``gold`` split to grow the hand-labelled evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decider.cli._common import configure_logging
from decider.schema import Record


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


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, default=Path("data/feedback/queue.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/feedback/labeled.jsonl"))
    p.add_argument("--split", default="train", choices=["train", "gold"])
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.panel import Panel
    from rich.pretty import Pretty

    console = Console()
    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["meta"].get("request_id", ""))
    entries = [
        json.loads(line)
        for line in args.queue.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    todo = [e for e in entries if e["request_id"] not in done]
    console.print(f"{len(todo)} queued requests ({len(done)} already labelled)")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    labelled = 0
    with args.out.open("a", encoding="utf-8") as out:
        try:
            for entry in todo:
                console.print(Panel(Pretty(entry["state"]), title=entry["request_id"][:12]))
                targets: dict[str, Any] = {}
                for name in entry["review"]:
                    question = entry["questions"][name]
                    console.print(
                        f"[bold]{name}[/] ({question['type']}): {question['instructions']}"
                    )
                    console.print(f"  model: {entry['answers'][name]}")
                    target = _ask_target(console, question)
                    if target is not None:
                        targets[name] = target
                if not targets:
                    continue
                record = Record(
                    id=f"feedback-{entry['request_id']}",
                    source="feedback",
                    family="feedback",
                    split=args.split,
                    state=entry["state"],
                    questions={k: entry["questions"][k] for k in targets},
                    targets=targets,
                    label_source="human:review",
                    meta={"request_id": entry["request_id"], "model": entry["model_version"]},
                )
                out.write(record.model_dump_json() + "\n")
                labelled += 1
        except KeyboardInterrupt:
            console.print("\nstopped")
    console.print(f"labelled {labelled} requests -> {args.out}")


if __name__ == "__main__":
    main()
