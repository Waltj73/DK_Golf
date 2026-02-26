# app.py — DK Golf: PGA Optimizer (Single Entry • Balanced Cash • Pro Mode)
# ------------------------------------------------------------
# Upload DK Salaries CSV + (optional) DataGolf performance CSV
# Builds 1–3 lineups with constraints and optional "Spike lineup":
#   Lineup #2: 2 studs (>= threshold) + 1 punt (< 7k)
#
# Fixes:
# - Auto-detect player name columns (no manual renaming)
# - Cleans NaN/inf to prevent PuLP objective crashes
# - Avoids KeyError/indentation issues
#
# Requirements (put in requirements.txt):
# streamlit
# pandas
# numpy
# pulp
# ------------------------------------------------------------

from __future__ import annotations

import io
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pulp import (
        LpProblem,
        LpMaximize,
        LpVariable,
        lpSum,
        LpStatus,
        PULP_CBC_CMD,
        value,
    )
except Exception:
    st.error("PuLP is not installed. Add `pulp` to your requirements.txt and redeploy.")
    st.stop()


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="DK Golf — PGA Optimizer", layout="wide")

st.title("DK Golf — PGA Optimizer (Single Entry • Balanced Cash • Pro Mode)")


# =========================
# HELPERS
# =========================
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first matching column name from candidates (case-insensitive),
    or None if none found.
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # direct match
    for cand in candidates:
        if cand in cols:
            return cand
    # case-insensitive match
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    # fuzzy contains match
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(x: str) -> str:
    """
    Normalize player name for joining across sources.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip().lower()

    # Remove accents-ish (best-effort without extra deps)
    s = (
        s.replace("’", "'")
        .replace("`", "'")
        .replace("–", "-")
        .replace("—", "-")
    )

    # Remove punctuation except space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    parts = s.split()
    # drop suffix if last token is suffix
    if parts and parts[-1] in _SUFFIXES:
        parts = parts[:-1]

    return " ".join(parts)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_numeric(df: pd.DataFrame, cols: List[str], fill: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = to_num(df[c])
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(fill)
    return df


def percentile_rank(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series([50.0] * len(s), index=s.index)
    # 0..1 then 0..100
    return s.rank(pct=True).fillna(0.5) * 100.0


@dataclass
class BuildResult:
    lineup: pd.DataFrame
    total_salary: int
    total_proj: float
    status: str


# =========================
# SIDEBAR: WORKFLOW
# =========================
with st.expander("What to upload / workflow", expanded=False):
    st.markdown(
        """
**You need:**
1) **DraftKings Salaries CSV** (the DK export for the tournament slate)
2) *(Optional but recommended)* **DataGolf Performance CSV** (your performance table export)

**Flow:**
- Upload files
- Tune (or leave defaults)
- Generate 1–3 lineups
- Download CSV

**Pro tip**
- Use **2 lineups**:
  - Lineup 1 = balanced/safe
  - Lineup 2 = spike (2 studs + 1 punt)
"""
    )


# =========================
# UPLOADS
# =========================
st.subheader("Upload Files")

colA, colB = st.columns(2)

with colA:
    dk_file = st.file_uploader("DraftKings Salaries CSV", type=["csv"])
with colB:
    dg_file = st.file_uploader("DataGolf Performance CSV (optional)", type=["csv"])

if not dk_file:
    st.info("Upload a DraftKings Salaries CSV to begin.")
    st.stop()

# Read DK
dk = pd.read_csv(dk_file)
dk = _clean_cols(dk)

# Detect DK columns
dk_name_col = detect_col(dk, ["Name", "Player", "Player Name", "Golfer", "player_name"])
dk_salary_col = detect_col(dk, ["Salary", "salary"])
dk_avg_col = detect_col(
    dk,
    [
        "AvgPointsPerGame",
        "DK AvgPointsPerGame",
        "DKAvgPointsPerGame",
        "AvgPoints",
        "Avg DK Points",
        "Avg DK",
    ],
)

# Validate required columns
missing = []
if not dk_name_col:
    missing.append("Name")
if not dk_salary_col:
    missing.append("Salary")

if missing:
    st.error(
        "Your DK CSV is missing required columns (or they weren't recognized).\n\n"
        f"Missing: {', '.join(missing)}\n\n"
        f"Found columns: {list(dk.columns)}"
    )
    st.stop()

dk = dk.rename(columns={dk_name_col: "name", dk_salary_col: "salary"})
dk["name_key"] = dk["name"].map(norm_name)
dk["salary"] = to_num(dk["salary"]).fillna(0).astype(int)

# If DK avg points exists, keep it; else create empty
if dk_avg_col:
    dk = dk.rename(columns={dk_avg_col: "dk_avg"})
    dk["dk_avg"] = to_num(dk["dk_avg"]).replace([np.inf, -np.inf], np.nan)
else:
    dk["dk_avg"] = np.nan

# Drop invalid names
dk = dk[dk["name_key"].astype(bool)].copy()
dk = dk.drop_duplicates(subset=["name_key"], keep="first").reset_index(drop=True)

# =========================
# OPTIONAL: READ DATAGOLF
# =========================
dg = None
dg_merged = False
if dg_file:
    dg = pd.read_csv(dg_file)
    dg = _clean_cols(dg)

    dg_name_col = detect_col(dg, ["Player", "Name", "player_name", "Golfer"])
    if not dg_name_col:
        st.warning(
            "DataGolf CSV uploaded, but no recognizable player name column was found. "
            "We’ll run DK-only this week."
        )
        dg = None
    else:
        dg = dg.rename(columns={dg_name_col: "player_name"})
        dg["name_key"] = dg["player_name"].map(norm_name)
        dg = dg[dg["name_key"].astype(bool)].copy()

        # Try to find useful DataGolf performance columns (best-effort)
        # Common possibilities across exports
        dg_total_sg_col = detect_col(
            dg,
            ["TOTAL", "SG: TOTAL", "Strokes Gained: Total", "True SG", "TRUE SG", "true_sg", "sg_total"],
        )
        dg_t2g_col = detect_col(dg, ["T2G", "SG: T2G", "sg_t2g", "tee_to_green"])
        dg_app_col = detect_col(dg, ["APP", "Approach", "SG: APP", "sg_app", "sg_approach"])
        dg_putt_col = detect_col(dg, ["PUTT", "Putting", "SG: PUTT", "sg_putt"])
        dg_ott_col = detect_col(dg, ["OTT", "Off-the-tee", "Off the Tee", "SG: OTT", "sg_ott"])
        dg_arg_col = detect_col(dg, ["ARG", "Around the Green", "SG: ARG", "sg_arg"])

        # Keep a compact DG table for merge
        keep = ["name_key"]
        rename_map = {}
        for col, new in [
            (dg_total_sg_col, "dg_sg_total"),
            (dg_t2g_col, "dg_sg_t2g"),
            (dg_app_col, "dg_sg_app"),
            (dg_putt_col, "dg_sg_putt"),
            (dg_ott_col, "dg_sg_ott"),
            (dg_arg_col, "dg_sg_arg"),
        ]:
            if col:
                keep.append(col)
                rename_map[col] = new

        dg_small = dg[keep].rename(columns=rename_map).copy()

        # numeric clean
        dg_small = safe_numeric(
            dg_small,
            [c for c in dg_small.columns if c.startswith("dg_")],
            fill=np.nan,
        )

        # Merge
        merged = dk.merge(dg_small, on="name_key", how="left")
        dg_merged = True
else:
    merged = dk.copy()

# If DG not merged, merged already defined
if not dg_merged:
    merged = dk.copy()

# =========================
# PROJECTION BASELINE
# =========================
st.subheader("Projection Baseline (leave defaults unless you want to tune)")

c1, c2, c3 = st.columns(3)
with c1:
    anchor_pts = st.slider("Salary anchor points (median salary golfer)", 10.0, 120.0, 55.0, 0.5)
with c2:
    slope_pts = st.slider("Salary slope (points per $1k)", 0.5, 10.0, 3.0, 0.1)
with c3:
    w_dk = st.slider("Weight on DK AvgPointsPerGame", 0.0, 1.0, 0.70, 0.05)

# Create salary-based baseline projection
median_salary = float(merged["salary"].median()) if len(merged) else 8000.0
merged["base_salary_proj"] = anchor_pts + slope_pts * ((merged["salary"] - median_salary) / 1000.0)

# Blend with DK avg if present
# If dk_avg is missing, fallback to salary proj
merged["base_proj"] = merged["base_salary_proj"]
has_dk_avg = merged["dk_avg"].notna()
merged.loc[has_dk_avg, "base_proj"] = (
    (1.0 - w_dk) * merged.loc[has_dk_avg, "base_salary_proj"] + w_dk * merged.loc[has_dk_avg, "dk_avg"]
)

# =========================
# CUT SAFETY + SE PROJ
# =========================
st.subheader("Single Entry Blend")

cut_available = any(c.startswith("dg_") for c in merged.columns)

cs_weight = st.slider("CutSafety weight (SE cash bias)", 0.0, 0.60, 0.25, 0.01)

# Compute CutSafety from DataGolf if available; else neutral
if cut_available:
    # Prefer total SG; else T2G; else App; else neutral
    cs_src = None
    for col in ["dg_sg_total", "dg_sg_t2g", "dg_sg_app", "dg_sg_ott", "dg_sg_putt", "dg_sg_arg"]:
        if col in merged.columns and merged[col].notna().sum() > 5:
            cs_src = col
            break

    if cs_src:
        merged["cut_safety"] = percentile_rank(merged[cs_src])
    else:
        merged["cut_safety"] = 50.0
else:
    merged["cut_safety"] = 50.0

# SE projection = base_proj + (CutSafety z-ish bump)
# Bump is small and centered so it doesn't dominate
merged["cut_safety_adj"] = (merged["cut_safety"] - 50.0) / 50.0  # -1..+1
merged["se_proj"] = merged["base_proj"] * (1.0 + cs_weight * merged["cut_safety_adj"])

# Value metric
merged["value_1k"] = merged["se_proj"] / (merged["salary"] / 1000.0)

# Clean numerics to prevent NaN in PuLP objective
merged = safe_numeric(
    merged,
    ["base_salary_proj", "base_proj", "cut_safety", "cut_safety_adj", "se_proj", "value_1k", "dk_avg"],
    fill=0.0,
)

# Show top table
st.markdown("### Top Players (by SE projection)")
st.dataframe(
    merged.sort_values("se_proj", ascending=False)[
        ["name", "salary", "base_proj", "se_proj", "value_1k", "cut_safety"]
    ].head(25),
    use_container_width=True,
)


# =========================
# LINEUP BUILDER SETTINGS
# =========================
st.subheader("Build Lineups (Balanced + Randomness + Overlap Control)")

ROSTER_SIZE = 6
SALARY_CAP = 50000

b1, b2, b3, b4 = st.columns(4)
with b1:
    n_lineups = st.slider("Lineups", 1, 3, 2)
with b2:
    base_rand = st.slider("Randomness % (Lineup #1)", 0.0, 25.0, 6.0, 0.5)
with b3:
    spike_rand = st.slider("Randomness % (Lineup #2)", 0.0, 25.0, 12.0, 0.5)
with b4:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026, step=1)

# Balanced constraints
cA, cB, cC, cD = st.columns(4)
with cA:
    min_total_salary = st.slider("Min total salary", 45000, 50000, 49300, 100)
with cB:
    min_avg_salary = st.slider("Min avg salary per golfer", 6500, 9500, 8000, 100)
with cC:
    max_under_7000_safe = st.slider("Lineup #1: max golfers under $7,000", 0, 3, 0, 1)
with cD:
    max_over_10500 = st.slider("Max golfers over $10,500", 0, 3, 1, 1)

# C-mode spike construction (Lineup #2)
st.markdown("### Spike Mode (Lineup #2 only)")
s1, s2, s3 = st.columns(3)
with s1:
    force_spike = st.toggle("Lineup #2: Force 2 studs + 1 punt", value=True)
with s2:
    stud_threshold = st.number_input("Stud threshold ($)", min_value=9000, max_value=12000, value=10000, step=100)
with s3:
    punt_threshold = st.number_input("Punt threshold ($)", min_value=5000, max_value=8000, value=7000, step=100)

# Guardrails
st.markdown("### Guardrails")
g1, g2, g3 = st.columns(3)
with g1:
    enable_guardrails = st.toggle("Enable CutSafety guardrails", value=True)
with g2:
    cs_min_floor = st.slider("CutSafety floor (Lineup #1 hard floor)", 0.0, 90.0, 60.0, 1.0)
with g3:
    cs_min_avg = st.slider("Min average CutSafety (both lineups)", 0.0, 90.0, 65.0, 1.0)

# Overlap control
max_overlap = st.slider("Max overlap between lineups (when building 2–3)", 0, 6, 4, 1)

# Manual controls
st.markdown("### Optional Manual Controls (fast pivots)")
m1, m2, m3 = st.columns(3)
with m1:
    lock_names = st.multiselect("Lock golfers", options=sorted(merged["name"].unique().tolist()))
with m2:
    exclude_names = st.multiselect("Exclude golfers", options=sorted(merged["name"].unique().tolist()))
with m3:
    bump_names = st.multiselect("Bump golfers (small +%)", options=sorted(merged["name"].unique().tolist()))

bump_pct = st.slider("Bump % (applied to 'Bump golfers')", 0.0, 20.0, 5.0, 0.5)

# Build button
build = st.button("Generate Lineups", type="primary")


# =========================
# OPTIMIZER
# =========================
def build_lineup(
    pool: pd.DataFrame,
    objective_col: str,
    min_total_salary_i: int,
    max_under_7000: int,
    force_spike_mode: bool,
    stud_threshold_i: int,
    punt_threshold_i: int,
    lineup_index: int,
    prev_lineups: List[set],
) -> BuildResult:
    """
    Build one lineup using PuLP.
    """
    pool = pool.copy().reset_index(drop=True)

    # Variables
    x = [LpVariable(f"x_{lineup_index}_{i}", cat="Binary") for i in range(len(pool))]

    prob = LpProblem(f"DK_Golf_LU_{lineup_index+1}", LpMaximize)

    # Objective
    obj_vals = pool[objective_col].astype(float).values
    # Final safety: ensure objective values are finite
    obj_vals = np.nan_to_num(obj_vals, nan=0.0, posinf=0.0, neginf=0.0)
    prob += lpSum(obj_vals[i] * x[i] for i in range(len(pool)))

    # Roster size
    prob += lpSum(x) == ROSTER_SIZE

    # Salary constraints
    salaries = pool["salary"].astype(int).values
    prob += lpSum(int(salaries[i]) * x[i] for i in range(len(pool))) <= SALARY_CAP
    prob += lpSum(int(salaries[i]) * x[i] for i in range(len(pool))) >= int(min_total_salary_i)

    # Min avg salary per golfer
    prob += lpSum(int(salaries[i]) * x[i] for i in range(len(pool))) >= int(min_avg_salary) * ROSTER_SIZE

    # Max over 10.5k
    prob += lpSum(x[i] for i in range(len(pool)) if int(salaries[i]) > 10500) <= int(max_over_10500)

    # Under 7k (safe lineup rule)
    prob += lpSum(x[i] for i in range(len(pool)) if int(salaries[i]) < int(punt_threshold_i)) <= int(max_under_7000)

    # Spike mode construction for lineup #2 (index 1)
    if lineup_index == 1 and force_spike_mode:
        # exactly 1 punt
        prob += lpSum(x[i] for i in range(len(pool)) if int(salaries[i]) < int(punt_threshold_i)) == 1
        # exactly 2 studs
        prob += lpSum(x[i] for i in range(len(pool)) if int(salaries[i]) >= int(stud_threshold_i)) == 2

    # Guardrails based on cut_safety
    if enable_guardrails:
        cs = pool["cut_safety"].astype(float).values
        cs = np.nan_to_num(cs, nan=50.0, posinf=50.0, neginf=50.0)

        # Hard floor for lineup #1 only (and for lineup #2 only if spike not forced)
        if not (lineup_index == 1 and force_spike_mode):
            prob += lpSum(x[i] for i in range(len(pool)) if float(cs[i]) < float(cs_min_floor)) == 0

        # Min average cut safety always
        prob += lpSum(float(cs[i]) * x[i] for i in range(len(pool))) >= float(cs_min_avg) * ROSTER_SIZE

    # Locks / excludes by name
    name_list = pool["name"].astype(str).tolist()
    name_to_idx = {}
    for i, nm in enumerate(name_list):
        name_to_idx.setdefault(nm, []).append(i)

    for nm in lock_names:
        if nm in name_to_idx:
            prob += lpSum(x[i] for i in name_to_idx[nm]) == 1

    for nm in exclude_names:
        if nm in name_to_idx:
            prob += lpSum(x[i] for i in name_to_idx[nm]) == 0

    # Overlap control with previous lineups
    # Each new lineup cannot share more than max_overlap with any previous lineup
    if prev_lineups:
        for j, prev in enumerate(prev_lineups):
            prev_idx = [i for i, nm in enumerate(name_list) if nm in prev]
            if prev_idx:
                prob += lpSum(x[i] for i in prev_idx) <= int(max_overlap)

    # Solve
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = LpStatus.get(prob.status, str(prob.status))
    if status != "Optimal":
        return BuildResult(
            lineup=pd.DataFrame(),
            total_salary=0,
            total_proj=0.0,
            status=status,
        )

    chosen = [i for i in range(len(pool)) if x[i].value() == 1]
    lu = pool.loc[chosen, ["name", "salary", "base_proj", "se_proj", "cut_safety"]].copy()

    # Total salary / proj
    tot_sal = int(lu["salary"].sum())
    tot_proj = float(lu["se_proj"].sum())

    # Sort display: high salary first
    lu = lu.sort_values("salary", ascending=False).reset_index(drop=True)

    return BuildResult(lineup=lu, total_salary=tot_sal, total_proj=tot_proj, status=status)


# =========================
# RUN BUILD
# =========================
if build:
    pool = merged.copy()

    # Apply bumps before randomness (small deterministic tweak)
    pool["bump_pct"] = 0.0
    if bump_names:
        pool.loc[pool["name"].isin(bump_names), "bump_pct"] = float(bump_pct)

    pool["final_base"] = pool["se_proj"] * (1.0 + pool["bump_pct"] / 100.0)

    # Ensure numeric cleanliness to prevent PuLP NaN/inf errors
    pool = safe_numeric(pool, ["final_base", "se_proj", "base_proj", "cut_safety"], fill=0.0)

    rng = np.random.default_rng(int(seed))

    results: List[BuildResult] = []
    prev_lineups: List[set] = []

    for k in range(int(n_lineups)):
        this_rand = float(base_rand) if k == 0 else float(spike_rand)

        noise = rng.normal(0.0, 1.0, size=len(pool))
        obj = pool["final_base"].astype(float).values * (1.0 + (this_rand / 100.0) * noise)

        # Keep objective finite
        obj = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)

        obj_col = f"obj_{k}"
        pool[obj_col] = obj

        # For lineup #1, enforce "max_under_7000_safe"
        # For lineup #2 spike mode, we'll allow under 7k (and possibly force it)
        max_under = int(max_under_7000_safe) if k == 0 else 6

        res = build_lineup(
            pool=pool,
            objective_col=obj_col,
            min_total_salary_i=int(min_total_salary),
            max_under_7000=max_under,
            force_spike_mode=bool(force_spike),
            stud_threshold_i=int(stud_threshold),
            punt_threshold_i=int(punt_threshold),
            lineup_index=k,
            prev_lineups=prev_lineups,
        )

        results.append(res)

        if not res.lineup.empty:
            prev_lineups.append(set(res.lineup["name"].tolist()))

    # Display
    st.markdown("---")
    st.subheader("Results")

    any_ok = False
    out_csv_rows = []

    for i, res in enumerate(results):
        st.markdown(f"### Lineup {i+1}")

        if res.status != "Optimal":
            st.error(f"Optimizer status: **{res.status}**. Try loosening constraints (min salary / overlap / locks).")
            continue

        any_ok = True
        a, b, c = st.columns(3)
        a.metric("Salary Used", f"${res.total_salary:,}")
        b.metric("Projected (SE)", f"{res.total_proj:.2f}")
        c.metric("Salary Left", f"${SALARY_CAP - res.total_salary:,}")

        st.dataframe(res.lineup, use_container_width=True)

        # Prepare export rows
        for _, row in res.lineup.iterrows():
            out_csv_rows.append(
                {
                    "lineup": i + 1,
                    "name": row["name"],
                    "salary": int(row["salary"]),
                    "base_proj": float(row["base_proj"]),
                    "se_proj": float(row["se_proj"]),
                    "cut_safety": float(row["cut_safety"]),
                }
            )

    if any_ok and out_csv_rows:
        out_df = pd.DataFrame(out_csv_rows)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Lineups CSV",
            data=csv_bytes,
            file_name="dk_golf_lineups.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption("Tip: If you keep seeing 'not optimal', reduce overlap, lower min salary, or remove locks.")
