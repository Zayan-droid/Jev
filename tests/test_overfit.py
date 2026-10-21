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
    generated = SyntheticGenerator(SyntheticConfig()).generate(64, "train", seed=0)
    # Hard labels only: tied (soft) targets put a floor under the reachable loss.
    records = [r for r in generated if r.label_kind == "hard"][:6]
    examples = [
        Example(t, record_to_example_targets(r))
        for t, r in zip(encoder.encode_many(records), records, strict=True)
    ]
    batch = collate(examples, limits.max_seq_len)
    assert batch.targets is not None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        loss, _ = decision_loss(model(batch), batch.targets, batch.option_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    # Uniform predictions score ~1.2 on this batch; memorising it must get far below that.
    # The bound is loose on purpose: the exact trajectory differs across BLAS backends.
    assert losses[0] > 0.8
    assert min(losses[-20:]) < losses[0] / 8
