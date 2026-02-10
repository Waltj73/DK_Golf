# app.py
import re
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DK Golf (PGA) — SE Cash Optimizer", layout="wide")
st.title("DK Golf — PGA Optimizer (Single Entry • Cash Consistency • Balanced)")

SALARY_CAP = 50000
ROSTER_SIZE = 6

# ---------------------------
# Helpers
# ---------------------------
def norm_name(s: str) -> str:
    """Normalize golfer names for merging across sources."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z\s\-']", "", s)  # keep letters/spaces/hyphen/apostrophe
    return s

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # exact matches
    for c in candidates:
        if c in df.columns:
            return c

    # case-insensitive matches
    cols_l = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_l:
            return cols_l[c.lower()]

    # flexible matching (handles player_name, player-name, etc.)
    def normalize(s):
        return s.lower().replace("_", "").replace(" ", "").replace("-", "")

    normalized_cols = {normalize(c): c for c in df.columns}

    for c in candidates:
        key = normalize(c)
        if key in normalized_cols:
            return normalized_cols[key]

    return None


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

# ---------------------------
# Uploads
# ---------------------------
st.subheader("Upload Files")
dk_file = st.file_uploader("1) DraftKings Salaries CSV", type=["csv"])
dg_file = st.file_uploader("2) DataGolf Performance Table CSV (the SG table)", type=["csv"])

if not dk_file:
    st.info("Upload DraftKings Salaries CSV to begin.")
    st.stop()

dk = pd.read_csv(dk_file)

# DK columns
dk_name = find_col(dk, ["Name", "Player Name", "Player"])
dk_salary = find_col(dk, ["Salary"])
dk_avg = find_col(dk, ["AvgPointsPerGame", "AvgPoints", "AvgPointsPerGame "])

if not dk_name or not dk_salary:
    st.error("DK file must include columns for Name and Salary.")
    st.stop()

dk[dk_name] = dk[dk_name].astype(str)
dk[dk_salary] = pd.to_numeric(dk[dk_salary], errors="coerce")
if dk_avg:
    dk[dk_avg] = pd.to_numeric(dk[dk_avg], errors="coerce")

dk = dk.dropna(subset=[dk_name, dk_salary]).copy()
dk = dk[dk[dk_salary] > 0].reset_index(drop=True)
dk["name_key"] = dk[dk_name].map(norm_name)

# ---------------------------
# Build DK baseline projection
# ---------------------------
st.subheader("Model Controls (defaults tuned for SE Cash)")

median_salary = float(dk[dk_salary].median())
target_median_points = st.slider("Salary-implied anchor points (median salary golfer)", 20.0, 90.0, 55.0, 0.5)
slope_per_1k = st.slider("Salary-implied slope (pts per $1k)", 1.0, 6.0, 3.0, 0.1)

b = slope_per_1k / 1000.0
a = target_median_points - b * median_salary
dk["SalaryImplied"] = (a + b * dk[dk_salary]).clip(lower=0)

use_dk_avg = (dk_avg is not None)
if use_dk_avg:
    w_avg = st.slider("Weight on DK AvgPointsPerGame", 0.0, 1.0, 0.70, 0.05)
    dk["BaseProj"] = w_avg * dk[dk_avg].fillna(dk["SalaryImplied"]) + (1 - w_avg) * dk["SalaryImplied"]
else:
    dk["BaseProj"] = dk["SalaryImplied"]

# ---------------------------
# DataGolf merge + CutSafety
# ---------------------------
cut_safety_available = False
if dg_file:
    dg = pd.read_csv(dg_file)

    dg_name = find_col(dg, ["Name",
    "Player",
    "player_name",
    "Player Name",
    "PLAYER"])
    # common column names for SG splits in performance tables
    col_app = find_col(dg, ["APP", "app", "SG_APP", "Approach", "sg_app"])
    col_ott = find_col(dg, ["OTT", "ott", "SG_OTT", "Off the Tee", "sg_ott"])
    col_arg = find_col(dg, ["ARG", "arg", "SG_ARG", "Around Green", "sg_arg"])
    col_putt = find_col(dg, ["PUTT", "putt", "SG_PUTT", "Putting", "sg_putt"])
    col_t2g = find_col(dg, ["T2G", "t2g", "SG_T2G", "Tee to Green", "sg_t2g"])
    col_total = find_col(dg, ["TOTAL", "total", "SG_TOTAL", "Total", "sg_total"])


    if not dg_name:
        st.warning("DataGolf file is missing a recognizable Name column. We'll run DK-only for now.")
    else:
        dg[dg_name] = dg[dg_name].astype(str)
        dg["name_key"] = dg[dg_name].map(norm_name)

        # Convert numeric cols
        for c in [col_app, col_ott, col_arg, col_putt, col_t2g, col_total]:
            if c:
                dg[c] = pd.to_numeric(dg[c], errors="coerce")

        # Merge (left join DK)
        merged = dk.merge(
            dg[["name_key"] + [c for c in [col_app, col_ott, col_arg, col_putt, col_t2g, col_total] if c]],
            on="name_key",
            how="left"
        )

        # Build CutSafety score from SG stats (z-scored for stability)
        # Fallback hierarchy: T2G+TOTAL if available else BallStrike proxy
        if col_t2g or col_app:
            z_app = zscore(merged[col_app]) if col_app else pd.Series(0, index=merged.index)
            z_ott = zscore(merged[col_ott]) if col_ott else pd.Series(0, index=merged.index)
            z_arg = zscore(merged[col_arg]) if col_arg else pd.Series(0, index=merged.index)
            z_putt = zscore(merged[col_putt]) if col_putt else pd.Series(0, index=merged.index)
            z_t2g = zscore(merged[col_t2g]) if col_t2g else pd.Series(np.nan, index=merged.index)
            z_total = zscore(merged[col_total]) if col_total else pd.Series(np.nan, index=merged.index)

            ball_strike = 0.55 * z_app + 0.35 * z_ott + 0.10 * z_arg
            if col_t2g and col_total:
                cut_safety = 0.70 * z_t2g + 0.30 * z_total
            elif col_t2g:
                cut_safety = z_t2g
            else:
                cut_safety = ball_strike

            # Penalize “putting-only” profiles for cash safety:
            # if putting z is high but ball_strike is poor, reduce safety slightly
            putting_only_penalty = np.where((z_putt > 0.8) & (ball_strike < -0.4), 0.25, 0.0)
            merged["CutSafetyZ"] = cut_safety - putting_only_penalty

            # Convert to 0–100 for display
            cs = merged["CutSafetyZ"]
            cs_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-9)
            merged["CutSafety"] = (100 * cs_norm).clip(0, 100)

            dk = merged
            cut_safety_available = True

# ---------------------------
# Build SE score
# ---------------------------
st.subheader("Single Entry Cash Scoring")

if cut_safety_available:
    se_cut_weight = st.slider("Weight on CutSafety (cash stability)", 0.0, 0.60, 0.25, 0.05)
else:
    st.info("No DataGolf file merged (or missing key SG columns). SE will run DK-only.")
    se_cut_weight = 0.0

# Normalize BaseProj into z-space to blend with CutSafety
dk["BaseProjZ"] = zscore(dk["BaseProj"])

if cut_safety_available:
    dk["CutSafetyZ"] = zscore(dk["CutSafety"])
    dk["SEScoreZ"] = (1 - se_cut_weight) * dk["BaseProjZ"] + se_cut_weight * dk["CutSafetyZ"]
else:
    dk["SEScoreZ"] = dk["BaseProjZ"]

# Convert SEScoreZ back to points-like scale
dk["SEProj"] = dk["BaseProj"] * (1 + 0.06 * dk["SEScoreZ"].clip(-2, 2) / 2.0)  # gentle ±6% adjustment

dk["Value_x1k"] = dk["SEProj"] / (dk[dk_salary] / 1000.0)

st.dataframe(
    dk.sort_values(["SEProj"], ascending=False)[
        [dk_name, dk_salary, "BaseProj", "SEProj", "Value_x1k"] + (["CutSafety"] if cut_safety_available else [])
    ].head(40),
    use_container_width=True
)

# ---------------------------
# Lineup construction controls (balanced, cash)
# ---------------------------
st.subheader("Build Lineups (Balanced Cash Defaults)")

num_lineups = st.slider("How many lineups?", 1, 3, 1, 1)
min_salary = st.number_input("Min total salary", min_value=0, max_value=SALARY_CAP, value=49300, step=100)

max_under_7k = st.slider("Max golfers under $7,000", 0, 3, 1, 1)
min_avg_salary = st.slider("Min average salary per golfer", 6500, 9500, 8000, 100)

max_over_10500 = st.slider("Max golfers over $10,500 (prevents stars/scrubs)", 0, 3, 1, 1)

# Optional manual bumps (simple, fast)
st.markdown("Optional quick bumps (keep it simple):")
all_names = dk[dk_name].tolist()
bump_players = st.multiselect("Select golfers to bump/penalize", options=all_names)
bump_map = {}
for p in bump_players:
    bump_map[p] = st.slider(f"{p} bump %", -25, 25, 0, 1)

dk["BumpPct"] = dk[dk_name].map(lambda n: bump_map.get(n, 0))
dk["FinalProj"] = dk["SEProj"] * (1 + dk["BumpPct"] / 100.0)

locks = st.multiselect("Lock golfers", options=all_names, default=[])
excludes = st.multiselect("Exclude golfers", options=[n for n in all_names if n not in locks], default=[])

run = st.button("Generate Lineup", type="primary")

if run:
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except Exception:
        st.error("PuLP not installed. Install with: `pip install pulp`")
        st.stop()

    base_df = dk[~dk[dk_name].isin(excludes)].copy().reset_index(drop=True)

    previous_sets = []
    all_lineups = []

    for k in range(num_lineups):
        prob = LpProblem(f"DK_Golf_SE_{k+1}", LpMaximize)
        x = [LpVariable(f"x_{k}_{i}", 0, 1, LpBinary) for i in range(len(base_df))]

        # Objective: maximize FinalProj
        prob += lpSum(base_df.loc[i, "FinalProj"] * x[i] for i in range(len(base_df)))

        # Exactly 6 golfers
        prob += lpSum(x[i] for i in range(len(base_df))) == ROSTER_SIZE

        # Salary cap + min salary
        prob += lpSum(base_df.loc[i, dk_salary] * x[i] for i in range(len(base_df))) <= SALARY_CAP
        prob += lpSum(base_df.loc[i, dk_salary] * x[i] for i in range(len(base_df))) >= min_salary

        # Balanced constraints
        prob += lpSum(x[i] for i in range(len(base_df)) if base_df.loc[i, dk_salary] < 7000) <= max_under_7k
        prob += lpSum(base_df.loc[i, dk_salary] * x[i] for i in range(len(base_df))) >= min_avg_salary * ROSTER_SIZE
        prob += lpSum(x[i] for i in range(len(base_df)) if base_df.loc[i, dk_salary] > 10500) <= max_over_10500

        # Locks
        for lk in locks:
            idxs = base_df.index[base_df[dk_name] == lk].tolist()
            if idxs:
                prob += x[idxs[0]] == 1

        # If generating multiple lineups, force differences (at least 2 golfers different)
        if k > 0:
            for prev in previous_sets:
                prob += lpSum(x[i] for i in prev) <= 4  # max overlap 4 of 6

        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if str(prob.status) != "1":
            st.error("No feasible lineup found. Relax constraints (min salary / avg salary / locks).")
            break

        chosen_idx = {i for i in range(len(base_df)) if x[i].value() == 1}
        previous_sets.append(chosen_idx)

        chosen = base_df.loc[list(chosen_idx)].copy()
        total_salary = int(chosen[dk_salary].sum())
        total_proj = float(chosen["FinalProj"].sum())

        chosen = chosen.sort_values("FinalProj", ascending=False)
        chosen["Lineup"] = k + 1

        st.markdown(f"### Lineup {k+1}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Salary", f"${total_salary:,}")
        c2.metric("Projected", f"{total_proj:.2f}")
        c3.metric("Salary Left", f"${SALARY_CAP - total_salary:,}")

        display_cols = [dk_name, dk_salary, "FinalProj", "BumpPct", "BaseProj"] + (["CutSafety"] if cut_safety_available else [])
        st.dataframe(chosen[display_cols], use_container_width=True)

        all_lineups.append(chosen[["Lineup"] + display_cols])

    if all_lineups:
        out = pd.concat(all_lineups, ignore_index=True)
        st.download_button(
            "Download Lineups CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="dk_golf_lineups.csv",
            mime="text/csv"
        )
