# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DK PGA Optimizer (1–3 Lineups)", layout="wide")
st.title("DraftKings PGA Optimizer — 1–3 Lineups (GPP / Single Entry)")

SALARY_CAP = 50000
ROSTER_SIZE = 6

uploaded = st.file_uploader(
    "Upload DK Salaries CSV (your export with AvgPointsPerGame)",
    type=["csv"]
)

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

if not uploaded:
    st.info("Upload your DraftKings PGA salary file to begin.")
    st.stop()

df = pd.read_csv(uploaded)

name_col = find_col(df, ["Name"])
salary_col = find_col(df, ["Salary"])
avg_col = find_col(df, ["AvgPointsPerGame"])

if not name_col or not salary_col:
    st.error("Your CSV must include at least: Name, Salary.")
    st.stop()

df[name_col] = df[name_col].astype(str)
df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
if avg_col:
    df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")

df = df.dropna(subset=[name_col, salary_col]).copy()
df = df[df[salary_col] > 0].reset_index(drop=True)

# ----------------------------
# 1) Build baseline projections
# ----------------------------
st.subheader("1) Baseline Projection Model")

st.markdown(
    "We’ll build a baseline projection using:\n"
    "- **DK AvgPointsPerGame** (if available)\n"
    "- **Salary-implied points** (market expectation)\n"
    "- Then you can add manual bumps for edge."
)

median_salary = float(df[salary_col].median())

# Salary-implied curve controls
target_median_points = st.slider(
    "Anchor: implied points for median salary golfer",
    20.0, 90.0, 55.0, 0.5
)
slope_per_1k = st.slider(
    "Slope (points per $1k salary)",
    1.0, 6.0, 3.2, 0.1
)
b = slope_per_1k / 1000.0
a = target_median_points - b * median_salary
df["SalaryImplied"] = (a + b * df[salary_col]).clip(lower=0)

if avg_col:
    st.caption("Your file includes AvgPointsPerGame ✅ — we can blend it with salary-implied points.")
    w_avg = st.slider("Weight on AvgPointsPerGame", 0.0, 1.0, 0.65, 0.05)
    w_sal = 1.0 - w_avg
    df["BaseProj"] = (w_avg * df[avg_col].fillna(df["SalaryImplied"])) + (w_sal * df["SalaryImplied"])
else:
    st.caption("No AvgPointsPerGame found — using salary-implied projection only.")
    df["BaseProj"] = df["SalaryImplied"]

# Ceiling proxy (PGA is volatile; cheap golfers swing harder)
def ceiling_multiplier(sal: float) -> float:
    if sal < 6500:  return 1.55
    if sal < 7500:  return 1.45
    if sal < 8500:  return 1.38
    if sal < 9500:  return 1.30
    if sal < 10500: return 1.24
    return 1.18

df["Ceiling"] = df.apply(lambda r: r["BaseProj"] * ceiling_multiplier(float(r[salary_col])), axis=1)
df["Value_x1k"] = df["BaseProj"] / (df[salary_col] / 1000.0)

# Risk proxy: cheaper = higher risk
sal_min, sal_max = float(df[salary_col].min()), float(df[salary_col].max())
df["RiskScore"] = 100 * (1 - (df[salary_col] - sal_min) / (sal_max - sal_min + 1e-9))

st.dataframe(
    df.sort_values(["Value_x1k", "BaseProj"], ascending=False)[
        [name_col, salary_col] + ([avg_col] if avg_col else []) + ["SalaryImplied", "BaseProj", "Ceiling", "Value_x1k", "RiskScore"]
    ].head(40),
    use_container_width=True
)

# ----------------------------
# 2) Manual bumps (your edge)
# ----------------------------
st.subheader("2) Manual Edge Layer (Bumps/Penalties)")

st.markdown(
    "This is where you create real edge for GPP/SE:\n"
    "- Weather wave / tee time advantage\n"
    "- Course fit\n"
    "- Injury / WD risk\n"
    "- Gut fades of mega-chalk\n"
)

all_names = df[name_col].tolist()
bump_players = st.multiselect("Select golfers to bump/penalize", options=all_names)

bump_map = {}
for p in bump_players:
    bump_map[p] = st.slider(f"{p} bump %", -40, 40, 0, 1)

df["BumpPct"] = df[name_col].map(lambda n: bump_map.get(n, 0))
df["AdjProj"] = df["BaseProj"] * (1 + df["BumpPct"] / 100.0)
df["AdjCeiling"] = df["Ceiling"] * (1 + df["BumpPct"] / 100.0)

mode = st.radio("Optimize for:", ["GPP (AdjCeiling)", "SE/Cash-ish (AdjProj)"], horizontal=True)
target_col = "AdjCeiling" if mode.startswith("GPP") else "AdjProj"

# ----------------------------
# 3) Build 1–3 lineups with diversification
# ----------------------------
st.subheader("3) Generate 1–3 Lineups (Diversified)")

num_lineups = st.slider("How many lineups?", 1, 3, 3, 1)
min_salary = st.number_input("Min total salary (avoid leaving too much)", min_value=0, max_value=SALARY_CAP, value=49200, step=100)

# Diversification: max overlap between lineup k and lineup k-1
# For 6 golfers, max overlap 4 means at least 2 different golfers.
max_overlap = st.slider("Max overlap between lineups", 0, 6, 4, 1)

locks = st.multiselect("Lock golfers (must include)", options=all_names, default=[])
excludes = st.multiselect("Exclude golfers", options=[n for n in all_names if n not in locks], default=[])

# Controlled randomness (small): adds noise to target to produce alternative but strong lineups
use_random = st.checkbox("Add small randomness (helps uniqueness)", value=True)
rand_strength = st.slider("Randomness strength (stdev % of projection)", 0.0, 12.0, 4.0, 0.5)

run = st.button("Build Lineups", type="primary")

if run:
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except Exception:
        st.error("PuLP not installed. Install with: `pip install pulp`")
        st.stop()

    base_df = df[~df[name_col].isin(excludes)].copy().reset_index(drop=True)

    # Apply locks existence check
    for lk in locks:
        if lk not in base_df[name_col].values:
            st.warning(f"Locked golfer not found after excludes: {lk}")

    previous_lineups = []  # store sets of indices
    lineup_outputs = []

    rng = np.random.default_rng(7)

    for k in range(num_lineups):
        opt_df = base_df.copy()

        # randomize objective a bit to create alternative lineups
        if use_random and rand_strength > 0:
            noise = rng.normal(loc=0.0, scale=(rand_strength / 100.0), size=len(opt_df))
            opt_df["Obj"] = opt_df[target_col] * (1 + noise)
        else:
            opt_df["Obj"] = opt_df[target_col]

        prob = LpProblem(f"DK_PGA_Lineup_{k+1}", LpMaximize)
        x = [LpVariable(f"x_{k}_{i}", 0, 1, LpBinary) for i in range(len(opt_df))]

        # Objective
        prob += lpSum(opt_df.loc[i, "Obj"] * x[i] for i in range(len(opt_df)))

        # Roster size
        prob += lpSum(x[i] for i in range(len(opt_df))) == ROSTER_SIZE

        # Salary
        prob += lpSum(opt_df.loc[i, salary_col] * x[i] for i in range(len(opt_df))) <= SALARY_CAP
        prob += lpSum(opt_df.loc[i, salary_col] * x[i] for i in range(len(opt_df))) >= min_salary

        # Locks
        for lk in locks:
            idxs = opt_df.index[opt_df[name_col] == lk].tolist()
            if idxs:
                prob += x[idxs[0]] == 1

        # Diversification constraint vs previous lineup
        # sum of overlap <= max_overlap
        for prev in previous_lineups:
            prob += lpSum(x[i] for i in prev) <= max_overlap

        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if str(prob.status) != "1":
            st.error(
                f"Could not find a feasible lineup for Lineup {k+1}. "
                f"Try lowering min salary, removing locks, or increasing max overlap."
            )
            break

        chosen_idx = {i for i in range(len(opt_df)) if x[i].value() == 1}
        previous_lineups.append(chosen_idx)

        chosen = opt_df.loc[list(chosen_idx)].copy()
        total_salary = int(chosen[salary_col].sum())
        total_score = float(chosen[target_col].sum())

        chosen = chosen.sort_values(target_col, ascending=False)
        chosen["Lineup"] = k + 1

        lineup_outputs.append((k + 1, total_salary, total_score, chosen[[ "Lineup", name_col, salary_col, "BaseProj", "BumpPct", target_col, "Ceiling", "RiskScore" ]]))

    if lineup_outputs:
        for (lid, tsal, tscore, ldf) in lineup_outputs:
            st.markdown(f"### Lineup {lid}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Salary", f"${tsal:,}")
            c2.metric("Optimized Score", f"{tscore:.2f}")
            c3.metric("Salary Left", f"${SALARY_CAP - tsal:,}")
            st.dataframe(ldf, use_container_width=True)

        # Download all lineups in one CSV
        out_df = pd.concat([x[3] for x in lineup_outputs], ignore_index=True)
        st.download_button(
            "Download All Lineups (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="dk_pga_lineups.csv",
            mime="text/csv"
        )
