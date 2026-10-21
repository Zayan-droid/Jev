from collections import Counter
from pathlib import Path

import pytest

from decider.data.build import BuildConfig, build_dataset
from decider.data.converters import MNLI, AGNews, Banking77, BoolQ, Clinc150, SciQ, YelpStars
from decider.data.dataset import EncodedSplit, MixtureSampler, TrainBatches
from decider.data.io import read_records, write_records
from decider.data.quality import deduplicate
from decider.data.synthetic import SyntheticConfig, SyntheticGenerator
from decider.packing import PackingLimits, RequestEncoder
from decider.tokenization import ByteTokenizer

# ----------------------------------------------------------------------------- converters


def test_banking77_converter() -> None:
    conv = Banking77(labels=["card_arrival", "lost_or_stolen_card", "top_up_failed"])
    record = conv.convert(
        {"text": "still waiting on my card", "label_text": "card_arrival"},
        "banking77-train-0000001",
        "train",
    )
    q = record.questions["intent"]
    assert q.type == "choice"
    assert q.criteria["card_arrival"] == "card arrival"
    assert record.targets["intent"] == {"card_arrival": 1.0}
    assert record.state == {"message": "still waiting on my card"}


def test_clinc_converter_keeps_gold_among_sampled_options() -> None:
    conv = Clinc150(labels=[f"intent_{i}" for i in range(150)] + ["oos"])
    record = conv.convert(
        {"text": "set an alarm", "intent": 150}, "clinc150-test-0000007", "heldout"
    )
    criteria = record.questions["intent"].criteria
    assert len(criteria) == Clinc150.n_options
    assert "oos" in criteria and record.targets["intent"] == {"oos": 1.0}
    again = conv.convert(
        {"text": "set an alarm", "intent": 150}, "clinc150-test-0000007", "heldout"
    )
    assert again == record, "conversion must be deterministic"


def test_ag_news_emits_choice_and_probe() -> None:
    record = AGNews().convert({"text": "Team wins cup", "label": 1}, "ag_news-train-1", "train")
    assert record.targets["topic"] == {"sports": 1.0}
    assert record.targets["is_probe"] in (0.0, 1.0)


def test_mnli_converter() -> None:
    row = {"premise": "A man plays guitar.", "hypothesis": "A person makes music.", "label": 0}
    record = MNLI().convert(row, "mnli-train-1", "train")
    assert record is not None
    target = record.targets["claim"]
    assert target == 1.0 or target == {"entailment": 1.0}
    assert MNLI().convert({**row, "label": -1}, "mnli-test-1", "test") is None


def test_boolq_yelp_sciq_converters() -> None:
    boolq = BoolQ().convert(
        {"question": "is water wet", "answer": True, "passage": "Water..."},
        "boolq-train-1",
        "train",
    )
    assert boolq.questions["answer"].instructions.endswith("Is water wet?")
    yelp = YelpStars().convert({"text": "meh", "label": 2}, "yelp-train-1", "train")
    assert "recommend" not in yelp.questions and yelp.targets["stars"] == [0, 0, 1, 0, 0]
    sciq = SciQ().convert(
        {
            "question": "Q?",
            "correct_answer": "x",
            "distractor1": "a",
            "distractor2": "b",
            "distractor3": "c",
            "support": "",
        },
        "sciq-test-1",
        "heldout",
    )
    gold = next(iter(sciq.targets["answer"]))
    assert sciq.questions["answer"].criteria[gold] == "x"


# ----------------------------------------------------------------------------- synthetic


def test_synthetic_records_are_valid_and_respect_heldout() -> None:
    cfg = SyntheticConfig(heldout_families=("dates",), heldout_templates=("arith.difference",))
    gen = SyntheticGenerator(cfg)
    train = list(gen.generate(400, "train", seed=0))
    held = list(gen.generate(200, "heldout", seed=0))
    assert {r.family for r in train} == {"inventory", "inventory_text", "arithmetic", "lookup"}
    assert "dates" in {r.family for r in held}
    train_templates = {t for r in train for t in r.meta["templates"]}
    assert "arith.difference" not in train_templates
    assert all(
        t in ("arith.difference",) or t.startswith("dates.")
        for r in held
        for t in r.meta["templates"]
    )


def test_synthetic_is_deterministic() -> None:
    gen = SyntheticGenerator(SyntheticConfig())
    a = [r.model_dump() for r in gen.generate(50, "train", seed=3)]
    b = [r.model_dump() for r in gen.generate(50, "train", seed=3)]
    assert a == b


def test_negation_pairs_have_complementary_targets() -> None:
    gen = SyntheticGenerator(SyntheticConfig(negation_rate=1.0))
    for record in gen.generate(100, "train", seed=0):
        pair = record.meta.get("negation_pair")
        if pair:
            assert record.targets[pair[0]] + record.targets[pair[1]] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- build


def test_build_dataset_end_to_end(tmp_path: Path) -> None:
    cfg = BuildConfig.from_dict(
        {
            "name": "t",
            "output_root": str(tmp_path),
            "synthetic": {
                "families": ["inventory", "arithmetic"],
                "heldout_templates": ["inv.most_common_color"],
            },
            "synthetic_sizes": {"train": 200, "val": 20, "test": 40, "heldout": 20},
        }
    )
    out = build_dataset(cfg)
    manifest = (out / "manifest.json").read_text()
    assert '"data_hash"' in manifest
    train = read_records(out / "train.jsonl.gz")
    test = read_records(out / "test.jsonl.gz")
    assert len(train) > 150 and len(test) > 30
    again = build_dataset(cfg)
    assert (again / "train.jsonl.gz").read_bytes() == (out / "train.jsonl.gz").read_bytes()


def test_deduplicate_removes_leakage() -> None:
    gen = SyntheticGenerator(SyntheticConfig(families=("arithmetic",)))
    records = list(gen.generate(20, "train", seed=0))
    leaked = [r.model_copy(update={"split": "test", "id": r.id + "-copy"}) for r in records[:5]]
    clean, report = deduplicate({"train": records, "test": leaked})
    assert clean["test"] == []
    assert report.leakage_removed["test"] == 5


def test_records_roundtrip(tmp_path: Path) -> None:
    records = list(SyntheticGenerator(SyntheticConfig()).generate(10, "train", seed=1))
    write_records(tmp_path / "x.jsonl.gz", records)
    assert read_records(tmp_path / "x.jsonl.gz") == records


# ----------------------------------------------------------------------------- sampling


def test_mixture_sampler_proportions_and_resume() -> None:
    sources = ["a"] * 900 + ["b"] * 100
    sampler = MixtureSampler(sources, alpha=0.0, seed=0)  # uniform over sources
    draws = [sources[next(sampler)] for _ in range(4000)]
    share_b = Counter(draws)["b"] / len(draws)
    assert 0.45 < share_b < 0.55

    state = sampler.state_dict()
    expected = [next(sampler) for _ in range(50)]
    restored = MixtureSampler(sources, alpha=0.0, seed=123)
    restored.load_state_dict(state)
    assert [next(restored) for _ in range(50)] == expected


def test_train_batches_respect_token_budget(tmp_path: Path) -> None:
    records = list(SyntheticGenerator(SyntheticConfig()).generate(200, "train", seed=0))
    write_records(tmp_path / "train.jsonl.gz", records)
    limits = PackingLimits(max_seq_len=1024, max_state_tokens=512)
    split = EncodedSplit.load(tmp_path, "train", RequestEncoder(ByteTokenizer(), limits))
    batches = TrainBatches(split, MixtureSampler(split.sources), limits, 4096, pad_id=256)
    for _ in range(5):
        batch, stats = next(batches)
        assert stats.tokens - max(split.lengths) < 4096
        assert 0 < stats.packing_efficiency <= 1
        assert batch.targets is not None
