import torch

from decider.training.losses import decision_loss


def _inputs(z_choice: torch.Tensor) -> tuple[dict, dict, dict]:
    logits = {
        "noul": torch.zeros(0),
        "choice": z_choice,
        "score": torch.zeros(0, 0),
    }
    targets = {
        "noul": torch.zeros(0),
        "choice": torch.tensor([[0.7, 0.3, 0.0, 0.0]]),
        "score": torch.zeros(0, 0),
    }
    mask = {
        "choice": torch.tensor([[True, True, True, False]]),
        "score": torch.zeros(0, 0, dtype=torch.bool),
    }
    return logits, targets, mask


def test_loss_is_minimised_at_the_target_distribution() -> None:
    """A proper scoring rule: optimising free logits recovers the (soft) target."""
    z = torch.zeros(1, 4, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.1)
    for _ in range(500):
        logits, targets, mask = _inputs(
            z.masked_fill(~torch.tensor([[1, 1, 1, 0]]).bool(), float("-inf"))
        )
        loss, _ = decision_loss(logits, targets, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    probs = torch.softmax(z[0, :3], dim=-1)
    torch.testing.assert_close(probs, torch.tensor([0.7, 0.3, 0.0]), atol=0.02, rtol=0)


def test_padding_gets_no_gradient_and_loss_is_finite() -> None:
    z = torch.zeros(1, 4, requires_grad=True)
    logits, targets, mask = _inputs(
        z.masked_fill(~torch.tensor([[1, 1, 1, 0]]).bool(), float("-inf"))
    )
    loss, parts = decision_loss(logits, targets, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and z.grad[0, 3] == 0
    assert "loss_choice" in parts


def test_noul_and_score_terms() -> None:
    logits = {
        "noul": torch.tensor([3.0, -3.0]),
        "choice": torch.zeros(0, 0),
        "score": torch.tensor([[2.0, 0.0, -2.0]]),
    }
    targets = {
        "noul": torch.tensor([1.0, 0.0]),
        "choice": torch.zeros(0, 0),
        "score": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    mask = {
        "choice": torch.zeros(0, 0, dtype=torch.bool),
        "score": torch.ones(1, 3, dtype=torch.bool),
    }
    loss, parts = decision_loss(logits, targets, mask)
    assert torch.isfinite(loss)
    assert {"loss_noul", "loss_score", "rps"} <= set(parts)
