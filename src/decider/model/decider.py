from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from decider.masks import decision_mask, sliding_window
from decider.model.config import ModelConfig
from decider.model.encoder import Encoder
from decider.model.heads import DecisionHeads
from decider.model.layers import KV
from decider.packing import TYPES, Batch


@dataclass
class StateCache:
    """Per-layer keys/values of an encoded state. Valid for any set of questions, because
    state tokens never attend to question tokens."""

    kv: list[KV]
    length: int

    def nbytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in self.kv)


def to_probabilities(
    logits: dict[str, torch.Tensor], log_temperature: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    probs: dict[str, torch.Tensor] = {}
    for i, t in enumerate(TYPES):
        z = logits[t]
        if log_temperature is not None:
            z = z / log_temperature[i].exp()
        probs[t] = torch.sigmoid(z) if t == "noul" else torch.softmax(z, dim=-1)
    return probs


def build_layer_masks(
    encoder: Encoder,
    cross_option_layers: int,
    doc_ids: torch.Tensor,
    question_ids: torch.Tensor,
    option_ids: torch.Tensor,
    position_ids: torch.Tensor,
    query_slice: slice | None = None,
) -> list[torch.Tensor]:
    n_layers = encoder.cfg.n_layers
    cache: dict[tuple[bool, bool], torch.Tensor] = {}
    masks = []
    for i in range(n_layers):
        key = (encoder.cfg.is_global(i), i >= n_layers - cross_option_layers)
        if key not in cache:
            is_global, cross = key
            mask = decision_mask(doc_ids, question_ids, option_ids, cross, query_slice)
            if not is_global:
                window = encoder.cfg.local_window
                mask = sliding_window(mask, position_ids, window, query_slice)
            cache[key] = mask
        masks.append(cache[key])
    return masks


class DeciderModel(nn.Module):
    log_temperature: torch.Tensor

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder)
        hidden = cfg.head_hidden or cfg.encoder.d_model
        self.heads = DecisionHeads(cfg.encoder.d_model, hidden, cfg.head_dropout)
        # Post-hoc temperature per question type (noul, choice, score); 0 means T = 1.
        self.register_buffer("log_temperature", torch.zeros(len(TYPES)))

    # ------------------------------------------------------------------ masks / forward

    def layer_masks(self, batch: Batch, query_slice: slice | None = None) -> list[torch.Tensor]:
        """One attention mask per encoder layer (at most four distinct tensors)."""
        return build_layer_masks(
            self.encoder,
            self.cfg.cross_option_layers,
            batch.doc_ids,
            batch.question_ids,
            batch.option_ids,
            batch.position_ids,
            query_slice,
        )

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        masks = self.layer_masks(batch)
        hidden, _ = self.encoder(batch.input_ids, batch.position_ids, batch.level_ids, masks)
        return self.heads(hidden.flatten(0, 1), batch.index, batch.option_mask)

    def probabilities(
        self, logits: dict[str, torch.Tensor], calibrated: bool = True
    ) -> dict[str, torch.Tensor]:
        return to_probabilities(logits, self.log_temperature if calibrated else None)

    # ------------------------------------------------------------------ temperatures

    def temperatures(self) -> dict[str, float]:
        return {t: float(self.log_temperature[i].exp()) for i, t in enumerate(TYPES)}

    def set_temperatures(self, temperatures: dict[str, float]) -> None:
        with torch.no_grad():
            for i, t in enumerate(TYPES):
                self.log_temperature[i] = torch.tensor(temperatures.get(t, 1.0)).log()

    # ------------------------------------------------------------------ state cache

    @torch.no_grad()
    def encode_state(self, state_ids: list[int]) -> StateCache:
        device = self.log_temperature.device
        ids = torch.tensor([state_ids], device=device)
        zeros = torch.zeros_like(ids)
        positions = torch.arange(len(state_ids), device=device).unsqueeze(0)
        masks = build_layer_masks(self.encoder, 0, zeros, zeros, zeros, positions)
        _, kv = self.encoder(ids, positions, zeros, masks, return_kv=True)
        return StateCache(kv=kv, length=len(state_ids))

    @torch.no_grad()
    def forward_and_cache(
        self, batch: Batch, state_length: int
    ) -> tuple[dict[str, torch.Tensor], StateCache]:
        """One full pass over a single-request batch that also returns the state's cache.

        The state's keys/values are the same inside a full pass as in ``encode_state`` (the
        state never attends to questions), so a cache miss costs one pass, not two.
        """
        if batch.input_ids.shape[0] != 1:
            raise ValueError("forward_and_cache expects exactly one request per batch")
        hidden, kv = self.encoder(
            batch.input_ids,
            batch.position_ids,
            batch.level_ids,
            self.layer_masks(batch),
            return_kv=True,
        )
        state_kv = [(k[:, :, :state_length].clone(), v[:, :, :state_length].clone()) for k, v in kv]
        logits = self.heads(hidden.flatten(0, 1), batch.index, batch.option_mask)
        return logits, StateCache(kv=state_kv, length=state_length)

    @torch.no_grad()
    def forward_cached(self, cache: StateCache, batch: Batch) -> dict[str, torch.Tensor]:
        """Answer the questions of a single-request batch against a cached state.

        ``batch`` is the usual unpacked layout (state tokens first); only the question tokens
        are run through the encoder.
        """
        if batch.input_ids.shape[0] != 1:
            raise ValueError("forward_cached expects exactly one request per batch")
        s = cache.length
        tail = slice(s, None)
        hidden, _ = self.encoder(
            batch.input_ids[:, tail],
            batch.position_ids[:, tail],
            batch.level_ids[:, tail],
            self.layer_masks(batch, query_slice=tail),
            prefix=cache.kv,
        )
        index = {t: (idx - s).clamp(min=0) for t, idx in batch.index.items()}
        return self.heads(hidden.flatten(0, 1), index, batch.option_mask)

    # ------------------------------------------------------------------ utilities

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
