"""Converters from public datasets to :class:`decider.schema.Record`.

A converter only maps rows; loading, sampling and splitting live in :mod:`decider.data.build`.
``convert`` is a pure function of the row, which keeps converters unit-testable without
downloads. Phrasings are picked deterministically from the row id, so a rebuild is
byte-identical.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from decider.schema import Record, Target

LABEL_SOURCE = "human"


def stable_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def humanize(label: str) -> str:
    return label.replace("_", " ").replace("/", " / ").strip()


class Converter(ABC):
    name: ClassVar[str]
    family: ClassVar[str]
    hf_path: ClassVar[str]
    hf_config: ClassVar[str | None] = None
    # Hugging Face split -> role. "train" rows are further divided into train/val/calib.
    splits: ClassVar[dict[str, str]]
    license: ClassVar[str]

    def setup(self, dataset: Any) -> None:  # noqa: B027 - optional hook
        """Read label names etc. from the loaded dataset before converting rows."""

    @abstractmethod
    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record | None: ...

    def record(
        self,
        rid: str,
        split: str,
        state: Any,
        questions: dict[str, Any],
        targets: dict[str, Target],
        **meta: Any,
    ) -> Record:
        return Record(
            id=rid,
            source=self.name,
            family=self.family,
            split=split,  # type: ignore[arg-type]
            state=state,
            questions=questions,
            targets=targets,
            label_kind="hard",
            label_source=LABEL_SOURCE,
            meta=meta,
        )


class Banking77(Converter):
    name = "banking77"
    family = "intent_routing"
    hf_path = "mteb/banking77"
    splits: ClassVar[dict[str, str]] = {"train": "train", "test": "test"}
    license = "CC-BY-4.0"

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        self.labels = list(labels) if labels is not None else []

    def setup(self, dataset: Any) -> None:
        # The mteb mirror stores label names as a column rather than a ClassLabel feature.
        if not self.labels:
            pairs = sorted(set(zip(dataset["label"], dataset["label_text"], strict=True)))
            self.labels = [text for _, text in pairs]

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        rng = stable_rng(self.name, rid)
        instructions = rng.choice(
            [
                "Which banking intent best describes `message`?",
                "Route `message` to the intent it expresses.",
                "What is the customer asking for in `message`?",
            ]
        )
        criteria = {label: humanize(label) for label in self.labels}
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
        return self.record(
            rid,
            split,
            {"message": row["text"]},
            {"intent": question},
            {"intent": {row["label_text"]: 1.0}},
        )


class Clinc150(Converter):
    """CLINC150 ("plus"): 150 intents across 10 domains plus out-of-scope.

    Used as a held-out intent dataset. Each record offers the gold intent and 19 distractors so
    the option set stays within the sequence budget.
    """

    name = "clinc150"
    family = "intent_routing"
    hf_path = "clinc/clinc_oos"
    hf_config = "plus"
    splits: ClassVar[dict[str, str]] = {"test": "heldout"}
    license = "CC-BY-3.0"
    n_options = 20

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        self.labels = list(labels) if labels is not None else []

    def setup(self, dataset: Any) -> None:
        self.labels = list(dataset.features["intent"].names)

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        rng = stable_rng(self.name, rid)
        gold = self.labels[int(row["intent"])]
        others = [label for label in self.labels if label != gold]
        options = [gold, *rng.sample(others, self.n_options - 1)]
        rng.shuffle(options)

        def describe(label: str) -> str:
            return "none of the supported intents apply" if label == "oos" else humanize(label)

        question = {
            "type": "choice",
            "instructions": "Which intent does the user express in `utterance`?",
            "criteria": {label: describe(label) for label in options},
        }
        return self.record(
            rid, split, {"utterance": row["text"]}, {"intent": question}, {"intent": {gold: 1.0}}
        )


class AGNews(Converter):
    name = "ag_news"
    family = "topic"
    hf_path = "fancyzhx/ag_news"
    splits: ClassVar[dict[str, str]] = {"train": "train", "test": "test"}
    license = "non-commercial research use"
    topics: ClassVar[dict[str, str]] = {
        "world": "international news, politics and conflicts",
        "sports": "sport results, athletes and competitions",
        "business": "companies, markets and the economy",
        "science_tech": "science, technology and the internet",
    }

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        rng = stable_rng(self.name, rid)
        names = list(self.topics)
        gold = names[int(row["label"])]
        probe = rng.choice(names)
        questions = {
            "topic": {
                "type": "choice",
                "instructions": "What is the main topic of `article`?",
                "criteria": dict(self.topics),
            },
            "is_probe": {
                "type": "noul",
                "instructions": f"Is `article` mainly about {self.topics[probe]}?",
            },
        }
        targets: dict[str, Target] = {"topic": {gold: 1.0}, "is_probe": float(probe == gold)}
        return self.record(rid, split, {"article": row["text"]}, questions, targets)


class MNLI(Converter):
    name = "mnli"
    family = "entailment"
    hf_path = "nyu-mll/glue"
    hf_config = "mnli"
    splits: ClassVar[dict[str, str]] = {"train": "train", "validation_matched": "test"}
    license = "OANC / CC-BY-SA-3.0 / CC-BY-3.0 (mixed, see GLUE)"
    relations: ClassVar[dict[str, str]] = {
        "entailment": "the text implies the claim is true",
        "neutral": "the text neither supports nor contradicts the claim",
        "contradiction": "the text implies the claim is false",
    }

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record | None:
        label = int(row["label"])
        if label < 0:
            return None
        relation = list(self.relations)[label]
        claim = row["hypothesis"].strip()
        rng = stable_rng(self.name, rid)
        if rng.random() < 0.5:
            question = {
                "type": "choice",
                "instructions": f"How does `text` relate to this claim: {claim}",
                "criteria": dict(self.relations),
            }
            target: Target = {relation: 1.0}
        else:
            question = {
                "type": "noul",
                "instructions": f"Does `text` establish that the following is true: {claim}",
            }
            target = float(relation == "entailment")
        return self.record(
            rid, split, {"text": row["premise"]}, {"claim": question}, {"claim": target}
        )


class BoolQ(Converter):
    name = "boolq"
    family = "reading_qa"
    hf_path = "google/boolq"
    splits: ClassVar[dict[str, str]] = {"train": "train", "validation": "test"}
    license = "CC-BY-SA-3.0"

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        q = row["question"].strip()
        instructions = f"According to `passage`: {q[0].upper()}{q[1:]}?"
        return self.record(
            rid,
            split,
            {"passage": row["passage"]},
            {"answer": {"type": "noul", "instructions": instructions}},
            {"answer": float(bool(row["answer"]))},
        )


class YelpStars(Converter):
    name = "yelp"
    family = "rating"
    hf_path = "Yelp/yelp_review_full"
    splits: ClassVar[dict[str, str]] = {"train": "train", "test": "test"}
    license = "Yelp Dataset Terms of Use (non-commercial)"
    levels: ClassVar[list[str]] = [
        "1 star: a very negative experience",
        "2 stars: mostly negative",
        "3 stars: mixed or average",
        "4 stars: mostly positive",
        "5 stars: an excellent experience",
    ]

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        stars = int(row["label"])
        questions: dict[str, Any] = {
            "stars": {
                "type": "score",
                "instructions": "How many stars did the author of `review` most likely give?",
                "levels": list(self.levels),
            }
        }
        targets: dict[str, Target] = {"stars": [float(i == stars) for i in range(5)]}
        if stars != 2:  # 3-star reviews have no clear recommend / not-recommend answer
            questions["recommend"] = {
                "type": "noul",
                "instructions": "Would the author of `review` recommend this business?",
            }
            targets["recommend"] = float(stars >= 3)
        return self.record(rid, split, {"review": row["text"]}, questions, targets)


class SciQ(Converter):
    """Science multiple choice; held out as an unseen task family."""

    name = "sciq"
    family = "science_qa"
    hf_path = "allenai/sciq"
    splits: ClassVar[dict[str, str]] = {"test": "heldout"}
    license = "CC-BY-NC-3.0"

    def convert(self, row: dict[str, Any], rid: str, split: str) -> Record:
        rng = stable_rng(self.name, rid)
        answers = [
            row["correct_answer"],
            row["distractor1"],
            row["distractor2"],
            row["distractor3"],
        ]
        rng.shuffle(answers)
        keys = "ABCD"
        criteria = {k: a for k, a in zip(keys, answers, strict=True)}
        gold = keys[answers.index(row["correct_answer"])]
        state = {"question": row["question"], "background": row.get("support") or ""}
        question = {
            "type": "choice",
            "instructions": "Which answer correctly answers `question`?",
            "criteria": criteria,
        }
        return self.record(rid, split, state, {"answer": question}, {"answer": {gold: 1.0}})


CONVERTERS: dict[str, type[Converter]] = {
    c.name: c for c in (Banking77, Clinc150, AGNews, MNLI, BoolQ, YelpStars, SciQ)
}
