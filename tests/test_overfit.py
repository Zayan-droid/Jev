"""The model can drive the loss of a fixed batch towards zero: gradients reach every head."""

import torch

from decider.data.dataset import record_to_example_targets
from decider.data.synthetic import SyntheticConfig, SyntheticGenerator
from decider.model import EncoderConfig, ModelConfig, build_model
from decider.packing import Example, PackingLimits, RequestEncoder, collate
from decider.training.losses import decision_loss


def test_overfits_a_single_batch() -> None:
    torch.manual_seed(0)
    model, tok = build_model(
        ModelConfig(
            encoder=EncoderConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128),
            cross_option_layers=1,
            head_dropout=0.0,
        )
    )
    limits = PackingLimits(max_seq_len=768, max_state_tokens=384)
    encoder = RequestEncoder(tok, limits)
    records = list(SyntheticGenerator(SyntheticConfig()).generate(8, "train", seed=0))
    examples = [
        Example(t, record_to_example_targets(r))
        for t, r in zip(encoder.encode_many(records), records, strict=True)
    ]
    batch = collate(examples, limits.max_seq_len)
    assert batch.targets is not None
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss, _ = decision_loss(model(batch), batch.targets, batch.option_mask)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Soft (tied) targets bound the reachable minimum away from zero; compare to the start.
    assert loss.item() < 0.1 * first
