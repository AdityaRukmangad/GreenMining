#!/usr/bin/env python3
"""
postProcess_extract.py
======================
Extract OpenFOAM field data into a CSV dataset compatible with the
GreenMining ML pipeline.

Output schema:
  Time, x, y, z, CH4, CO, H2, Temperature, Velocity, Pressure, Risk, Scenario

Risk thresholds (volume fraction):
  SAFE    : CH4 < 0.010  AND CO < 0.0001 AND H2 < 0.010
  WARNING : CH4 < 0.025  AND CO < 0.0005 AND H2 < 0.020
  DANGER  : otherwise

Usage:
  python3 postProcess_extract.py [--scenario N] [--case .] [--out output.csv]

Requirements: numpy, pandas
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def classify_risk(ch4, co, h2):
    danger = (ch4 >= 0.025) | (co >= 0.0005) | (h2 >= 0.020)
    warning = (~danger) & ((ch4 >= 0.010) | (co >= 0.0001) | (h2 >= 0.010))
    return np.where(danger, "DANGER", np.where(warning, "WARNING", "SAFE"))


_LIST_SCALAR_RE = re.compile(
    r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)",
    re.DOTALL,
)
_UNIFORM_SCALAR_RE = re.compile(r"internalField\s+uniform\s+([\d.eE+\-]+)")
_LIST_VECTOR_RE = re.compile(
    r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)",
    re.DOTALL,
)
_UNIFORM_VECTOR_RE = re.compile(
    r"internalField\s+uniform\s+\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)"
)


def _parse_scalar(text, n):
    m = _LIST_SCALAR_RE.search(text)
    if m:
        return np.fromstring(m.group(2), sep="\n")[:n]
    m = _UNIFORM_SCALAR_RE.search(text)
    if m:
        return np.full(n, float(m.group(1)))
    raise ValueError("Cannot parse scalar internalField")


def _parse_vector(text, n):
    m = _LIST_VECTOR_RE.search(text)
    if m:
        triplets = re.findall(
            r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)", m.group(2)
        )
        arr = np.array([(float(a), float(b), float(c)) for a, b, c in triplets])
        return arr[:n, 0], arr[:n, 1], arr[:n, 2]
    m = _UNIFORM_VECTOR_RE.search(text)
    if m:
        return (np.full(n, float(m.group(1))),
                np.full(n, float(m.group(2))),
                np.full(n, float(m.group(3))))
    raise ValueError("Cannot parse vector internalField")


def read_field(path, n, kind="scalar"):
    text = Path(path).read_text(errors="replace")
    return _parse_scalar(text, n) if kind == "scalar" else _parse_vector(text, n)


def count_cells(fp):
    text = Path(fp).read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*\n\s*(\d+)", text)
    return int(m.group(1)) if m else None


def find_time_dirs(case_dir):
    dirs = []
    for p in case_dir.iterdir():
        if p.is_dir() and p.name != "0":
            try:
                dirs.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(dirs)


def read_cell_centres(time_dirs, n):
    for _, td in reversed(time_dirs):
        if (td / "Cx").exists():
            return (read_field(td / "Cx", n),
                    read_field(td / "Cy", n),
                    read_field(td / "Cz", n))
    raise FileNotFoundError(
        "Cx/Cy/Cz not found. Run simulation to completion or: "
        "postProcess -func writeCellCentres"
    )


SCALAR_FIELDS = {"CH4": "CH4", "CO": "CO", "H2": "H2",
                 "T": "Temperature", "p_rgh": "Pressure"}


def extract(case_dir, scenario, out_path):
    print(f"Case: {case_dir}")
    time_dirs = find_time_dirs(case_dir)
    if not time_dirs:
        print("No time directories found. Run the solver first.")
        sys.exit(1)
    print(f"Time steps: {len(time_dirs)}")

    n = None
    for _, td in time_dirs[:1]:
        for fname in ("CH4", "CO", "H2", "T", "p_rgh"):
            fp = td / fname
            if fp.exists():
                n = count_cells(fp)
                if n:
                    break
    if not n:
        print("Cannot determine cell count.")
        sys.exit(1)
    print(f"Cells: {n}")

    cx, cy, cz = read_cell_centres(time_dirs, n)
    records = []

    for t, td in time_dirs:
        row = {"Time": t, "x": cx, "y": cy, "z": cz}
        for fname, col in SCALAR_FIELDS.items():
            fp = td / fname
            row[col] = read_field(fp, n) if fp.exists() else np.zeros(n)
        fp_u = td / "U"
        if fp_u.exists():
            ux, uy, uz = read_field(fp_u, n, "vector")
            row["Velocity"] = np.sqrt(ux**2 + uy**2 + uz**2)
        else:
            row["Velocity"] = np.zeros(n)
        df_t = pd.DataFrame(row)
        df_t["Risk"] = classify_risk(df_t["CH4"].values, df_t["CO"].values, df_t["H2"].values)
        df_t["Scenario"] = scenario
        records.append(df_t)

    df = pd.concat(records, ignore_index=True)
    for col in ("x", "y", "z"):
        df[col] = df[col].round(3)

    cols = ["Time", "x", "y", "z", "CH4", "CO", "H2",
            "Temperature", "Velocity", "Pressure", "Risk", "Scenario"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_path, index=False)

    print(f"\nExported {len(df):,} rows -> {out_path}")
    print(df["Risk"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default=".")
    p.add_argument("--scenario", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    case_dir = Path(args.case).resolve()
    out = Path(args.out) if args.out else case_dir / f"mine_cfd_output_scenario{args.scenario}.csv"
    extract(case_dir, args.scenario, out)
