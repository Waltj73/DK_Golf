# app.py
# DK Golf — PGA Optimizer (Single Entry • Cash Consistency • Balanced)
# Upload:
#  1) DraftKings Salaries CSV (must include: Name, Salary; optionally AvgPointsPerGame)
#  2) DataGolf Performance Table CSV (expects player_name + *_true columns, but robust to variations)
#
# Requires:
#   pip install streamlit pandas numpy pulp

import re
import numpy as np
import pandas as pd
import streamlit as st

SALARY_CAP = 50000
ROSTER_SIZE = 6

# =========================
# Helpers
# =========================
def norm_name(s: str) -> str:
    """Normalize names to improve merges between DK and DataGolf."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    # keep letters/spaces/hyphen/apostrophe
    s = re.sub(r"[^a-z\s\-']", "", s)
    return s

def normalize_colname(c: str) -> str:
    """Normalize column names to match across exports."""
    c = str(c).strip().lower()
    c = c.replace(" ", "_").replace("-", "_")
    c = re.sub(r"__+", "_", c)
    return c

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def pick_col_by_norm(df: pd.DataFrame, options_norm: list[str]) -> str | None:
    """
    Find the first column in df whose normalized name matches any normalized option.
    options_norm should already be normalized via normalize_colname().
    """
    norm_map = {normalize_colname(c): c for c in df.columns}
    for opt in options_norm:
        if opt in norm_map:
            return norm_map[opt]
    return None

def add_debug(st_container, dk: pd.DataFrame, dg: pd.DataFrame | None, debug: bool):
    if not debug:
        return
    with st_container:
        st.markdown("### Debug")
        st.write("DK columns:", list(dk.columns))
        if dg is not None:
            st.write("DataGolf columns:", list(dg.columns))
        st.write("DK rows:", len(dk))
        if dg is not None:
            st.write("DataGolf rows:", len(dg))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DK Golf — PGA Optimizer (SE Cash • Balanced)", layout="wide")
st.title("DK Golf — PGA Optimizer (Single Entry • Cash Consistency • Balanced)")

with st.expander("Requirements / Setup", expanded=False):
    st.code("pip install streamlit pandas numpy pulp", language="bash")
    st.markdown(
        "- Upload DK salaries CSV.\n"
        "- Upload DataGolf performance table CSV.\n"
        "- Click **Generate Lineup**.\n"
        "\n"
        "**Tip:** If PuLP isn't installed, the optimizer can't run."
    )

debug_mode = st.checkbox("Debug mode", value=False)

st.subheader("Upload Files")
dk_file = st.file_uploader("DraftKings Salaries CSV", type=["csv"])
dg_file = st.file_uploader("DataGolf Performance Table CSV (SG table)", type=["csv"])

if not dk_file:
    st.info("Upload your DraftKings Salaries CSV to begin.")
    st.stop()

# =========================
# Load DK
# =========================
dk = pd.read_csv(dk_file)

# Normalize DK columns
dk.columns = [c.strip() for c in dk.columns]

dk_name_col = None
for candidate in ["Name", "Player Name", "Player"]:
    if candidate in dk.columns:
        dk_name_col = candidate
        break
if dk_name_col is None:
    # fallback by normalized matching
    dk_name_col = pick_col_by_norm(dk, [normalize_colname("name"), normalize_colname("player_name"), normalize_colname("player")])

dk_salary_col = None
for candidate in ["Salary"]:
    if candidate in dk.columns:
        dk_salary_col = candidate
        break
if dk_salary_col is None:
    dk_salary_col = pick_col_by_norm(dk, [normalize_colname("salary")])

dk_avg_col = None
for candidate in ["AvgPointsPerGame", "AvgPoints", "AvgPointsPerGame "]:
    if candidate in dk.columns:
        dk_avg_col = candidate
        break
if dk_avg_col is None:
    dk_avg_col = pick_col_by_norm(dk, [normalize_colname("avgpointspergame"), normalize_colname("avg_points_per_game")])

if not dk_name_col or not dk_salary_col:
    st.error("DK file must include at least columns for Name and Salary.")
    st.stop()

dk[dk_name_col] = dk[dk_name_col].astype(str)
dk[dk_salary_col] = pd.to_numeric(dk[dk_salary_col], errors="coerce")
if dk_avg_col:
    dk[dk_avg_col] = pd.to_numeric(dk[dk_avg_col], errors="coerce")

dk = dk.dropna(subset=[dk_name_col, dk_salary_col]).copy()
dk = dk[dk[dk_salary_col] > 0].reset_index(drop=True)

dk["name_key"] = dk[dk_name_col].map(norm_name)

add_debug(st, dk, None, debug_mode)

# =========================
# Load DataGolf (optional)
# =========================
dg = None
cut_safety_available = False

if dg_file:
    dg = pd.read_csv(dg_file)
    # normalize DataGolf column names for robust matching
    dg.columns = [normalize_colname(c) for c in dg.columns]

    # Determine DG name column
    dg_name_col = pick_col_by_norm(
        dg,
        [normalize_colname(x) for x in ["player_name", "name", "player", "golfer", "playername"]]
    )

    if dg_name_col:
        dg[dg_name_col] = dg[dg_name_col].astype(str)
        dg["name_key"] = dg[dg_name_col].map(norm_name)

        # Detect SG columns (handles *_true, sg_*, and plain labels)
        col_app = pick_col_by_norm(dg, [normalize_colname(x) for x in ["app_true", "sg_app", "app"]])
        col_ott = pick_col_by_norm(dg, [normalize_colname(x) for x in ["ott_true", "sg_ott", "ott"]])
        col_arg = pick_col_by_norm(dg, [normalize_colname(x) for x in ["arg_true", "sg_arg", "arg"]])
        col_putt = pick_col_by_norm(dg, [normalize_colname(x) for x in ["putt_true", "sg_putt", "putt"]])
        col_t2g = pick_col_by_norm(dg, [normalize_colname(x) for x in ["t2g_true", "sg_t2g", "t2g"]])
        col_total = pick_col_by_norm(dg, [normalize_colname(x) for x in ["total_true", "sg_total", "total"]])

        if debug_mode:
            st.write("Detected DG columns:", {
                "dg_name_col": dg_name_col,
                "app": col_app, "ott": col_ott, "arg": col_arg, "putt": col_putt, "t2g": col_t2g, "total": col_total
            })

        # Keep only needed columns
        keep_cols = ["name_key"] + [c for c in [col_putt, col_arg, col_app, col_ott, col_t2g, col_total] if c]
        dg_small = dg[keep_cols].copy()

        # Coerce numeric
        for c in [col_putt, col_arg, col_app, col_ott, col_t2g, col_total]:
            if c and c in dg_small.columns:
                dg_small[c] = pd.to_numeric(dg_small[c], errors="coerce")

        # Merge
        dk = dk.merge(dg_small, on="name_key", how="left")

        # Compute CutSafety if we have enough columns
        needed = [col_app, col_ott, col_t2g, col_total]
        if all(c is not None and c in dk.columns for c in needed):
            z_app = zscore(dk[col_app])
            z_ott = zscore(dk[col_ott])
            z_t2g = zscore(dk[col_t2g])
            z_total = zscore(dk[col_total])

            z_arg = zscore(dk[col_arg]) if (col_arg and col_arg in dk.columns) else pd.Series(0, index=dk.index)
            z_putt = zscore(dk[col_putt]) if (col_putt and col_putt in dk.columns) else pd.Series(0, index=dk.index)

            ball_strike = 0.55 * z_app + 0.35 * z_ott + 0.10 * z_arg
            cut_safety_z = 0.70 * z_t2g + 0.30 * z_total

            # Penalize "putting-only" spikes (less reliable for cut safety)
            putting_only_penalty = np.where((z_putt > 0.8) & (ball_strike < -0.4), 0.25, 0.0)

            dk["CutSafetyZ"] = cut_safety_z - putting_only_penalty

            # Scale to 0–100
            cs = dk["CutSafetyZ"]
            cs_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-9)
            dk["CutSafety"] = (100 * cs_norm).clip(0, 100)

            cut_safety_available = True
        else:
            dk["CutSafety"] = 0.0
            cut_safety_available = False
    else:
        st.warning("DataGolf file is missing a recognizable player name column (e.g., player_name). Running DK-only.")
        dk["CutSafety"] = 0.0
        cut_safety_available = False
else:
    dk["CutSafety"] = 0.0
    cut_safety_available = False

add_debug(st, dk, dg, debug_mode)

# =========================
# Baseline projection model
# =========================
st.subheader("Model Controls (defaults tuned for SE Cash)")

median_salary = float(dk[dk_salary_col].median())
target_median_points = st.slider("Salary-implied anchor points (median salary golfer)", 20.0, 90.0, 55.0, 0.5)
slope_per_1k = st.slider("Salary-implied slope (pts per $1k)", 1.0, 6.0, 3.0, 0.1)

b = slope_per_1k / 1000.0
a = target_median_points - b * median_salary
dk["SalaryImplied"] = (a + b * dk[dk_salary_col]).clip(lower=0)

if dk_avg_col:
    w_avg = st.slider("Weight on DK AvgPointsPerGame", 0.0, 1.0, 0.70, 0.05)
    dk["BaseProj"] = w_avg * dk[dk_avg_col].fillna(dk["SalaryImplied"]) + (1 - w_avg) * dk["SalaryImplied"]
else:
    dk["BaseProj"] = dk["SalaryImplied"]

# =========================
# Single Entry Cash scoring
# =========================
st.subheader("Single Entry Cash Scoring")

if cut_safety_available:
    se_cut_weight = st.slider("Weight on CutSafety (cash stability)", 0.0, 0.60, 0.25, 0.05)
else:
    st.info("No DataGolf CutSafety available. Running DK-only.")
    se_cut_weight = 0.0

dk["BaseProjZ"] = zscore(dk["BaseProj"])
if cut_safety_available:
    dk["CutSafetyZ2"] = zscore(dk["CutSafety"])
    dk["SEScoreZ"] = (1 - se_cut_weight) * dk["BaseProjZ"] + se_cut_weight * dk["CutSafetyZ2"]
else:
    dk["SEScoreZ"] = dk["BaseProjZ"]

# Gentle +/- adjustment to base projection (keeps things stable for SE)
dk["SEProj"] = dk["BaseProj"] * (1 + 0.06 * dk["SEScoreZ"].clip(-2, 2) / 2.0)
dk["Value_x1k"] = dk["SEProj"] / (dk[dk_salary_col] / 1000.0)

# Display top table
show_cols = [dk_name_col, dk_salary_col, "BaseProj", "SEProj", "Value_x1k"]
if cut_safety_available:
    show_cols += ["CutSafety"]

st.dataframe(
    dk.sort_values("SEProj", ascending=False)[show_cols].head(50),
    use_container_width=True
)

# =========================
# Build lineups (Balanced Cash defaults)
# =========================
st.subheader("Build Lineups (Balanced Cash Defaults)")

num_lineups = st.slider("How many lineups?", 1, 3, 1, 1)
min_salary = st.number_input("Min total salary", min_value=0, max_value=SALARY_CAP, value=49300, step=100)

max_under_7k = st.slider("Max golfers under $7,000", 0, 3, 1, 1)
min_avg_salary = st.slider("Min average salary per golfer", 6500, 9500, 8000, 100)
max_over_10500 = st.slider("Max golfers over $10,500 (prevents stars/scrubs)", 0, 3, 1, 1)

# Optional bumps
st.markdown("**Optional quick bumps (keep it simple):**")
all_names = dk[dk_name_col].tolist()

bump_players = st.multiselect("Select golfers to bump/penalize", options=all_names)
bump_map: dict[str, int] = {}
for p in bump_players:
    bump_map[p] = st.slider(f"{p} bump %", -25, 25, 0, 1)

dk["BumpPct"] = dk[dk_name_col].map(lambda n: bump_map.get(n, 0))
dk["FinalProj"] = dk["SEProj"] * (1 + dk["BumpPct"] / 100.0)

locks = st.multiselect("Lock golfers", options=all_names, default=[])
excludes = st.multiselect("Exclude golfers", options=[n for n in all_names if n not in locks], default=[])

run = st.button("Generate Lineup", type="primary")

if run:
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except Exception:
        st.error("PuLP not installed. Install with: pip install pulp")
        st.stop()

    pool = dk[~dk[dk_name_col].isin(excludes)].copy().reset_index(drop=True)

    if len(pool) < ROSTER_SIZE:
        st.error("Not enough golfers left after exclusions.")
        st.stop()

    previous_sets: list[set[int]] = []
    lineup_frames: list[pd.DataFrame] = []

    for k in range(num_lineups):
        prob = LpProblem(f"DK_GOLF_SE_{k+1}", LpMaximize)
        x = [LpVariable(f"x_{k}_{i}", 0, 1, LpBinary) for i in range(len(pool))]

        # Objective
        prob += lpSum(pool.loc[i, "FinalProj"] * x[i] for i in range(len(pool)))

        # Exactly 6 golfers
        prob += lpSum(x[i] for i in range(len(pool))) == ROSTER_SIZE

        # Salary cap + min salary
        prob += lpSum(pool.loc[i, dk_salary_col] * x[i] for i in range(len(pool))) <= SALARY_CAP
        prob += lpSum(pool.loc[i, dk_salary_col] * x[i] for i in range(len(pool))) >= min_salary

        # Balanced constraints
        prob += lpSum(x[i] for i in range(len(pool)) if pool.loc[i, dk_salary_col] < 7000) <= max_under_7k
        prob += lpSum(pool.loc[i, dk_salary_col] * x[i] for i in range(len(pool))) >= min_avg_salary * ROSTER_SIZE
        prob += lpSum(x[i] for i in range(len(pool)) if pool.loc[i, dk_salary_col] > 10500) <= max_over_10500

        # Locks
        for lk in locks:
            idxs = pool.index[pool[dk_name_col] == lk].tolist()
            if idxs:
                prob += x[idxs[0]] == 1

        # Diversify if multiple lineups: max overlap 4 of 6 (>=2 different golfers)
        if k > 0:
            for prev in previous_sets:
                prob += lpSum(x[i] for i in prev) <= 4

        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if str(prob.status) != "1":
            st.error("No feasible lineup found. Relax constraints (min salary / avg salary / locks).")
            break

        chosen_idx = {i for i in range(len(pool)) if x[i].value() == 1}
        previous_sets.append(chosen_idx)

        chosen = pool.loc[list(chosen_idx)].copy()
        total_salary = int(chosen[dk_salary_col].sum())
        total_proj = float(chosen["FinalProj"].sum())

        chosen = chosen.sort_values("FinalProj", ascending=False)
        chosen["Lineup"] = k + 1

        st.markdown(f"### Lineup {k+1}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Salary", f"${total_salary:,}")
        c2.metric("Projected", f"{total_proj:.2f}")
        c3.metric("Salary Left", f"${SALARY_CAP - total_salary:,}")

        disp_cols = ["Lineup", dk_name_col, dk_salary_col, "FinalProj", "BumpPct", "BaseProj"]
        if cut_safety_available:
            disp_cols += ["CutSafety"]
        st.dataframe(chosen[disp_cols], use_container_width=True)

        lineup_frames.append(chosen[disp_cols])

    if lineup_frames:
        out = pd.concat(lineup_frames, ignore_index=True)
        st.download_button(
            "Download Lineups CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="dk_golf_lineups.csv",
            mime="text/csv",
        )
