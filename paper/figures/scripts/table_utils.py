"""Formatting helpers for the paper tables (mean +- SD and paired differences, LaTeX)."""

from __future__ import annotations

import statistics as st
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1]


def ms(xs, bold: bool = False) -> str:
    """Mean +- sample SD over seeds, in math mode."""
    xs = list(xs)
    body = f"{st.mean(xs):.1f} \\pm {st.stdev(xs):.1f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def diffs(a: dict, b: dict) -> list[float]:
    """Paired differences a - b over seeds; both must cover the same seeds."""
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    return [a[k] - b[k] for k in sorted(a)]


def paired(a: dict, b: dict) -> str:
    """Signed paired difference a - b, mean +- SD over seeds, in math mode."""
    d = diffs(a, b)
    m = st.mean(d)
    sign = "" if abs(m) < 0.05 else ("+" if m > 0 else "-")  # no "-0.0"
    return f"${sign}{abs(m):.1f} \\pm {st.stdev(d):.1f}$"


def write_table(name: str, body: str):
    path = FIG_DIR / f"{name}.tex"
    path.write_text(body)
    print(f"Saved: {path}")
