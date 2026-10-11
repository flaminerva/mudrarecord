"""Compare the upstream IMU decoder with the 53-byte record decoder on the
same packets of a stream-csv recording made with --include-raw-bytes.

    python evidence/compare_decoders.py evidence/still_5s.csv [OUT_DIR]

Writes fig_a_abs_acc.png and fig_b_record_norm_hist.png to OUT_DIR (default:
the current directory). The upstream decoder is protocol.decode_imu_packet,
unchanged since upstream commit 7a7bce3; the new one is
protocol.split_imu_records.
"""
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mudrarecord import protocol  # noqa: E402

LSB_PER_G = 16384.0              # nominal, ±2 g
FIGSIZE, DPI = (12, 5), 200      # 2400 x 1000 px


def imu_packets(path):
    """[(host time s, raw bytes)] of the IMU packets, each once."""
    out, last = [], None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["stream"] == "snc":
                continue
            key = (row["timestamp_ns"], row["raw_hex"])
            if key != last:
                out.append((int(row["timestamp_ns"]) / 1e9, bytes.fromhex(row["raw_hex"])))
                last = key
    t0 = out[0][0]
    return [(t - t0, raw) for t, raw in out]


def caption(fig, text):
    """One-line caption, wrapped so it stays inside the figure."""
    fig.text(0.5, 0.01, textwrap.fill(text, 120), ha="center", va="bottom", fontsize=11)


def main(csv_path, out_dir="."):
    packets = imu_packets(csv_path)
    name, out = Path(csv_path).name, Path(out_dir)
    dt = float(np.median(np.diff([t for t, _ in packets]))) / 8   # 8 samples per packet
    up_t, up_g, new_t, new_g, norms = [], [], [], [], {}
    for t, raw in packets:
        for k, s in enumerate(protocol.decode_imu_packet(raw)):
            up_t.append(t + k * dt)
            up_g.append(np.linalg.norm(s) / LSB_PER_G)
        for rec in protocol.split_imu_records(raw):
            norms.setdefault(rec.tag, []).append(np.mean([np.linalg.norm(s) for s in rec.samples]))
            if rec.kind == "acc":
                for k, s in enumerate(rec.samples):
                    new_t.append(t + k * dt)
                    new_g.append(np.linalg.norm(s) / LSB_PER_G)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax.plot(new_t, new_g, ".", ms=3, color="#1e5ac8",
            label=f"new decoder (split_imu_records, tag 0x0a): {len(new_g)} samples")
    ax.plot(up_t, up_g, ".", ms=3, color="#c81e1e",
            label=f"upstream decoder 7a7bce3 (decode_imu_packet): {len(up_g)} samples")
    ax.axhline(1.0, ls="--", color="k", lw=1, label="1 g")
    ax.set_xlabel("time since first IMU packet (s)")
    ax.set_ylabel("|acc| (g, 16384 LSB/g)")
    ax.set_ylim(0, 1.3)
    ax.legend(loc="center right")
    caption(fig, f"{name}: the same {len(packets)} IMU packets decoded by both decoders; "
                 "|acc| of every sample, 8 samples per packet spread over the packet interval.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "fig_a_abs_acc.png", facecolor="white")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    bins = np.logspace(0, 5, 101)
    for tag, colour in ((0x0A, "#1e5ac8"), (0x04, "#1e963c")):
        if tag in norms:
            ax.hist(norms[tag], bins=bins, color=colour, alpha=0.8,
                    label=f"tag byte 0x{tag:02x}: {len(norms[tag])} records")
    other = sum(len(v) for k, v in norms.items() if k not in (0x0A, 0x04))
    ax.set_xscale("log")
    ax.set_xlabel("mean |(x, y, z)| of the 8 int16 triplets in a 53-byte record (LSB)")
    ax.set_ylabel("records")
    ax.legend(title=f"other tag bytes: {other} records")
    caption(fig, f"{name}: every 106-byte IMU packet split into two 53-byte records "
                 "[tag][8 x 3 int16 LE][u32 timestamp]; histogram per tag byte.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "fig_b_record_norm_hist.png", facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:3])
