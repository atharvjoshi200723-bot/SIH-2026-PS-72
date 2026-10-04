"""
make_synthetic.py — CLI script to generate synthetic storm data.

Run via:
    make synthetic
or:
    python scripts/make_synthetic.py --config configs/config.yaml

Output files are written to data/synthetic/ as event_NNNN.npy files.
Each file is a float32 array shaped (frames_per_event, H, W), values in [0, 1].

This data is SYNTHETIC — it is generated from moving Gaussian blobs and does
NOT represent real meteorological observations. Every downstream step that
uses this data labels it clearly as synthetic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make sure `src` is importable when run as a script ───────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.synthetic import SyntheticGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg.paths.synthetic_root)

    log.info(
        "Generating %d synthetic events on a %dx%d grid...",
        cfg.synthetic.num_events,
        cfg.grid.height,
        cfg.grid.width,
    )

    gen = SyntheticGenerator(cfg.synthetic, cfg.grid)
    events = gen.generate()
    gen.save(events, output_dir)

    # Quick sanity check before we declare success.
    saved = sorted(output_dir.glob("event_*.npy"))
    assert len(saved) == len(events), "Save count mismatch!"

    sample = saved[0]
    arr = __import__("numpy").load(sample)
    log.info(
        "Done. %d files in %s. Sample shape: %s, min=%.3f, max=%.3f",
        len(saved),
        output_dir,
        arr.shape,
        float(arr.min()),
        float(arr.max()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic storm data.")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()
    main(args.config)
