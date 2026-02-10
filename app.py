# app.py
# DK Golf — PGA Optimizer (Single Entry • Balanced Cash • Pro Mode)
# What this version adds (your request "C"):
#   ✅ Balanced constraints (min salary, max punts, cap expensive studs, min avg salary)
#   ✅ Multi-lineup generator (1–3 lineups)
#   ✅ Randomness control (for uniqueness / SE GPP pivots)
#   ✅ Max overlap control across generated lineups
#   ✅ DK + DataGolf robust merge + CutSafety
#   ✅ NaN/inf-safe optimizer (won’t crash if some players don’t merge)
#
# Run:
#   streamlit run app.py
#
# Install:
#   pip install streamlit pandas numpy pulp

import re
import numpy as np
import pandas as pd
import streamlit as st
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

SALARY_CAP = 50000
ROSTER_SIZE = 6

# -------------------------
# Helpers
# -------------------------
def norm_name(s: str) -> str:
    s = str(s).strip().lower()
    # handle "Last, First"
    if "," in s:
        last, first = s.split(",", 1)
        s = first.strip() + " " + last.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z\s\-']", "", s)
    return s

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("+", "_")
    )
    return df

def find_col(df: pd.DataFrame, options: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in cols:
            return cols[opt.lower()]
    return None

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="DK Golf — SE Balanced Pro", layout="wide")
st.title("DK Golf — PGA Optimizer (Single Entry • Balanced Cash • Pro Mode)")

with st.expander("What to upload / workflow", expanded=False):
    st.markdown(
        "- Upload **DraftKings Salaries CSV** (standard export)\n"
        "- Upload **DataGolf Performance CSV** (the performance table export)\n"
        "- Tune **Balanced Constraints** + **Randomness**\n"
        "- Click **Generate Lineups** (1–3)\n"
    )

st.subheader("Upload Files")
dk_file = st.file_uploader("DraftKings Salaries CSV", type=["csv"])
dg_file = st.file_uploader("DataGolf Performance CSV", type=["csv"])

if not dk_file:
    st.stop()

# -------------------------
# Load DK
# -------------------------
dk = pd.read_csv(dk_file)
dk = normalize_cols(dk)

name_col = find_col(dk, ["name", "player", "name_id"])
salary_col = find_col(dk, ["salary"])
avg_col = find_col(dk, ["avgpointspergame", "avg_points_per_game", "avg_points"])

if name_col is None or salary_col is None:
    st.error("Could not detect DK columns. Your DK file must include at least Name and Salary.")
    st.stop()

dk[name_col] = dk[name_col].astype(str)
dk[salary_col] = pd.to_numeric(dk[salary_col], errors="coerce")
if avg_col:
    dk[avg_col] = pd.to_numeric(dk[avg_col], errors="coerce")

dk = dk.dropna(subset=[name_col, salary_col]).copy()
dk = dk[dk[salary_col] > 0].reset_index(drop=True)

dk["name_key"] = dk[name_col].map(norm_name)

# -------------------------
# Base projection (DK + salary curve)
# -------------------------
st.subheader("Projection Baseline (leave defaults unless you want to tune)")
median_salary = float(dk[salary_col].median())

target_median_points = st.slider("Salary anchor points (median salary golfer)", 20.0, 90.0, 55.0, 0.5)
slope_per_1k = st.slider("Salary slope (points per $1k)", 1.0, 6.0, 3.0, 0.1)

b = slope_per_1k / 1000.0
a = target_median_points - b * median_salary
dk["salary_proj"] = (a + b * dk[salary_col]).clip(lower=0)

if avg_col:
    w_avg = st.slider("Weight on DK AvgPointsPerGame", 0.0, 1.0, 0.70, 0.05)
    dk["base_proj"] = w_avg * dk[avg_col].fillna(dk["salary_proj"]) + (1 - w_avg) * dk["salary_proj"]
else:
    dk["base_proj"] = dk["salary_proj"]

# -------------------------
# Load DataGolf + merge + CutSafety
# -------------------------
cut_available = False
dk["cut_safety"] = 0.0

if dg_file:
    dg = pd.read_csv(dg_file)
    dg = normalize_cols(dg)

    dg_name = find_col(dg, ["player_name", "name", "player"])
    if dg_name is None:
        st.warning("Could not detect player name column in DataGolf file. Running DK-only.")
    else:
        dg[dg_name] = dg[dg_name].astype(str)
        dg["name_key"] = dg[dg_name].map(norm_name)

        # support DataGolf columns like *_true
        col_app = find_col(dg, ["app_true", "sg_app", "app"])
        col_ott = find_col(dg, ["ott_true", "sg_ott", "ott"])
        col_arg = find_col(dg, ["arg_true", "sg_arg", "arg"])
        col_putt = find_col(dg, ["putt_true", "sg_putt", "putt"])
        col_t2g = find_col(dg, ["t2g_true", "sg_t2g", "t2g"])
        col_total = find_col(dg, ["total_true", "sg_total", "total"])

        keep = ["name_key"] + [c for c in [col_app, col_ott, col_arg, col_putt, col_t2g, col_total] if c]
        dg_small = dg[keep].copy()
        dg_small = safe_numeric(dg_small, [col_app, col_ott, col_arg, col_putt, col_t2g, col_total])

        dk = dk.merge(dg_small, on="name_key", how="left")

        # compute cut safety if we have core columns
        if all(c is not None and c in dk.columns for c in [col_app, col_ott, col_t2g, col_total]):
            z_app = zscore(dk[col_app])
            z_ott = zscore(dk[col_ott])
            z_t2g = zscore(dk[col_t2g])
            z_total = zscore(dk[col_total])

            z_arg = zscore(dk[col_arg]) if col_arg and col_arg in dk.columns else pd.Series(0, index=dk.index)
            z_putt = zscore(dk[col_putt]) if col_putt and col_putt in dk.columns else pd.Series(0, index=dk.index)

            ball_strike = 0.55 * z_app + 0.35 * z_ott + 0.10 * z_arg
            cut_z = 0.70 * z_t2g + 0.30 * z_total

            # penalize "putting-only" profiles for cash stability
            penalty = np.where((z_putt > 0.8) & (ball_strike < -0.4), 0.25, 0.0)
            dk["cut_safety_z"] = cut_z - penalty

            cs = dk["cut_safety_z"]
            dk["cut_safety"] = 100 * (cs - cs.min()) / (cs.max() - cs.min() + 1e-9)
            dk["cut_safety"] = dk["cut_safety"].clip(0, 100)

            cut_available = True
        else:
            st.warning("DataGolf loaded but missing key SG columns (APP/OTT/T2G/TOTAL). Running DK-only.")

# -------------------------
# SE projection blend
# -------------------------
st.subheader("Single Entry Blend")
if cut_available:
    cut_weight = st.slider("CutSafety weight (SE cash bias)", 0.0, 0.60, 0.25, 0.05)
else:
    cut_weight = 0.0
    st.info("CutSafety unavailable (DG not merged or missing stats). SE will run DK-only.")

dk["proj_z"] = zscore(dk["base_proj"])
dk["cut_z2"] = zscore(dk["cut_safety"]) if cut_available else pd.Series(0, index=dk.index)

# Gentle adjustment; keeps DK baseline intact
blend_z = (1 - cut_weight) * dk["proj_z"] + cut_weight * dk["cut_z2"]
dk["se_proj"] = dk["base_proj"] * (1 + 0.06 * blend_z.clip(-2, 2) / 2.0)

# Clean numeric issues (prevents solver crash)
dk = dk.replace([np.inf, -np.inf], np.nan)
dk = dk.dropna(subset=["se_proj", salary_col, name_col]).copy()

# -------------------------
# Optional bumps + locks/excludes
# -------------------------
st.subheader("Optional Manual Controls (fast pivots)")
all_names = dk[name_col].tolist()

locks = st.multiselect("Lock golfers (optional)", options=all_names, default=[])
excludes = st.multiselect("Exclude golfers (optional)", options=[n for n in all_names if n not in locks], default=[])

bump_players = st.multiselect("Bump / Penalize golfers (optional)", options=[n for n in all_names if n not in excludes])
bump_map = {}
for p in bump_players:
    bump_map[p] = st.slider(f"{p} bump %", -25, 25, 0, 1)

dk["bump_pct"] = dk[name_col].map(lambda n: bump_map.get(n, 0))
dk["final_base"] = dk["se_proj"] * (1 + dk["bump_pct"] / 100.0)

# -------------------------
# Balanced Constraints + Multi-lineup + Randomness
# -------------------------
st.subheader("Build Lineups (Balanced + Randomness + Overlap Control)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    num_lineups = st.slider("Lineups", 1, 3, 1, 1)
with c2:
    rand_strength = st.slider("Randomness %", 0.0, 20.0, 6.0, 0.5)
with c3:
    max_overlap = st.slider("Max overlap (for lineup 2/3)", 0, 6, 4, 1)
with c4:
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026, step=1)

st.markdown("### Balanced Constraints")
d1, d2, d3, d4 = st.columns(4)
with d1:
    min_total_salary = st.number_input("Min total salary", min_value=0, max_value=SALARY_CAP, value=49300, step=100)
with d2:
    max_under_7k = st.slider("Max golfers under $7,000", 0, 3, 1, 1)
with d3:
    max_over_10500 = st.slider("Max golfers over $10,500", 0, 3, 1, 1)
with d4:
    min_avg_salary = st.slider("Min avg salary per golfer", 6500, 9500, 8000, 100)

run = st.button("Generate Lineups", type="primary")

# -------------------------
# Optimizer
# -------------------------
if run:
    pool = dk.copy()
    if excludes:
        pool = pool[~pool[name_col].isin(excludes)].copy()

    # Ensure locks exist in pool
    missing_locks = [lk for lk in locks if lk not in set(pool[name_col])]
    if missing_locks:
        st.error(f"Locked golfers not found after exclusions: {missing_locks}")
        st.stop()

    # Final cleanup for solver
    pool = pool.replace([np.inf, -np.inf], np.nan)
    pool = pool.dropna(subset=["final_base", salary_col, name_col]).copy()
    pool = pool[pool[salary_col] > 0].reset_index(drop=True)

    if len(pool) < ROSTER_SIZE:
        st.error("Not enough golfers in pool after cleaning/exclusions.")
        st.stop()

    rng = np.random.default_rng(int(seed))
    previous_sets: list[set[int]] = []
    out_frames: list[pd.DataFrame] = []

    for k in range(int(num_lineups)):
        # add per-lineup randomness
        noise = rng.normal(0.0, 1.0, size=len(pool))
        pool[f"obj_{k}"] = pool["final_base"] * (1.0 + (rand_strength / 100.0) * noise)

        prob = LpProblem(f"DK_Golf_Lineup_{k+1}", LpMaximize)
        x = [LpVariable(f"x_{k}_{i}", 0, 1, LpBinary) for i in range(len(pool))]

        # Objective
        prob += lpSum(pool.loc[i, f"obj_{k}"] * x[i] for i in range(len(pool)))

        # Roster size
        prob += lpSum(x[i] for i in range(len(pool))) == ROSTER_SIZE

        # Salary cap + min salary
        prob += lpSum(pool.loc[i, salary_col] * x[i] for i in range(len(pool))) <= SALARY_CAP
        prob += lpSum(pool.loc[i, salary_col] * x[i] for i in range(len(pool))) >= float(min_total_salary)

        # Balanced constraints
        prob += lpSum(x[i] for i in range(len(pool)) if pool.loc[i, salary_col] < 7000) <= int(max_under_7k)
        prob += lpSum(x[i] for i in range(len(pool)) if pool.loc[i, salary_col] > 10500) <= int(max_over_10500)
        prob += lpSum(pool.loc[i, salary_col] * x[i] for i in range(len(pool))) >= float(min_avg_salary) * ROSTER_SIZE

        # Locks
        for lk in locks:
            idxs = pool.index[pool[name_col] == lk].tolist()
            if idxs:
                prob += x[idxs[0]] == 1

        # Overlap control (for lineup 2/3)
        if k > 0:
            for prev in previous_sets:
                prob += lpSum(x[i] for i in prev) <= int(max_overlap)

        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if str(prob.status) != "1":
            st.error(
                "No feasible lineup found. Try relaxing constraints:\n"
                "- lower Min total salary\n"
                "- lower Min avg salary\n"
                "- allow more under $7k\n"
                "- reduce locks\n"
            )
            break

        chosen_idx = {i for i in range(len(pool)) if x[i].value() == 1}
        previous_sets.append(chosen_idx)

        lineup = pool.loc[list(chosen_idx)].copy()
        lineup["lineup"] = k + 1
        lineup = lineup.sort_values("final_base", ascending=False)

        total_salary = int(lineup[salary_col].sum())
        total_proj = float(lineup["final_base"].sum())

        st.markdown(f"## Lineup {k+1}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Salary Used", f"${total_salary:,}")
        m2.metric("Projected (Final)", f"{total_proj:.2f}")
        m3.metric("Salary Left", f"${SALARY_CAP - total_salary:,}")

        show_cols = [name_col, salary_col, "final_base", "bump_pct", "base_proj", "se_proj"]
        if cut_available:
            show_cols.append("cut_safety")
        st.dataframe(lineup[show_cols], use_container_width=True)

        out_frames.append(lineup[["lineup"] + show_cols])

    if out_frames:
        out = pd.concat(out_frames, ignore_index=True)
        st.download_button(
            "Download Lineups CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="dk_golf_lineups.csv",
            mime="text/csv",
        )

# -------------------------
# Player table
# -------------------------
st.subheader("Top Players (by SE projection)")
display_cols = [name_col, salary_col, "base_proj", "se_proj", "final_base"]
if cut_available:
    display_cols.append("cut_safety")

st.dataframe(
    dk.sort_values("final_base", ascending=False)[display_cols].head(50),
    use_container_width=True
)
