"""``decider-export``: export a bundle's model to ONNX and verify it against PyTorch.

    decider-export --bundle runs/<name>/<ts>/bundle         writes bundle/model.onnx

Verification runs both models on CPU over requests of several shapes and fails if the logits
differ by more than ``--tolerance``; it also reports the latency of both on this machine.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from decider.cli._common import configure_logging

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default: <bundle>/model.onnx")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args(argv)

    import numpy as np
    import torch

    from decider.bundle import load_bundle
    from decider.evaluation.latency import synthetic_request
    from decider.export import OnnxDecider, export_onnx, max_abs_difference
    from decider.packing import Example, RequestEncoder, collate

    bundle = load_bundle(args.bundle, "cpu")
    out = args.out or args.bundle / "model.onnx"
    export_onnx(bundle.model, bundle.tokenizer, bundle.limits, out)
    runner = OnnxDecider(out, bundle.model.log_temperature)
    encoder = RequestEncoder(bundle.tokenizer, bundle.limits)

    worst = 0.0
    for state_tokens, n_questions in (
        (32, 1),
        (128, 4),
        (min(384, bundle.limits.max_state_tokens), 8),
    ):
        request = synthetic_request(bundle.tokenizer, state_tokens, n_questions)
        batch = collate([Example(encoder.encode(request))], bundle.limits.max_seq_len, pack=False)
        with torch.inference_mode():
            reference = bundle.model(batch)
        worst = max(worst, max_abs_difference(reference, runner(batch)))

        timings: dict[str, list[float]] = {"torch": [], "onnxruntime": []}
        for _ in range(args.repeats):
            for name, fn in (("torch", bundle.model), ("onnxruntime", runner)):
                start = time.perf_counter()
                with torch.inference_mode():
                    fn(batch)
                timings[name].append((time.perf_counter() - start) * 1000)
        log.info(
            "state %d tokens, %d questions: torch %.1f ms, onnxruntime %.1f ms (CPU, p50)",
            state_tokens,
            n_questions,
            float(np.median(timings["torch"])),
            float(np.median(timings["onnxruntime"])),
        )
    log.info("max |logit difference| = %.2e (tolerance %.0e)", worst, args.tolerance)
    if not worst <= args.tolerance:
        raise SystemExit(f"exported model disagrees with PyTorch: {worst:.2e} > {args.tolerance}")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
