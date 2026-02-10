# DK Golf — PGA Optimizer (Single Entry • Cash Consistency • Balanced)
# Run: streamlit run app.py
# Requires: pip install streamlit pandas numpy pulp

import re
import numpy as np
import pandas as pd
import streamlit as st

SALARY_CAP = 50000
ROSTER_SIZE = 6

# ---------- Name handling ----------
def norm_name(s):
    s = str(s).strip().lower()
    if "," in s:
        last, first = s.split(",", 1)
        s = first.strip() + " " + last.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z\s\-']", "", s)
    return s


def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def normalize_cols(df):
    df.columns = (
        df.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def find_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("DK Golf — PGA Optimizer (Single Entry • Balanced Cash)")

st.subheader("Upload Files")
dk_file = st.file_uploader("DraftKings Salaries CSV", type=["csv"])
dg_file = st.file_uploader("DataGolf Performance CSV", type=["csv"])

if not dk_file:
    st.stop()

# ---------- Load DK ----------
dk = pd.read_csv(dk_file)
dk = normalize_cols(dk)

name_col = find_col(dk, ["name", "player"])
salary_col = find_col(dk, ["salary"])
avg_col = find_col(dk, ["avgpointspergame"])

dk["name_key"] = dk[name_col].map(norm_name)
dk[salary_col] = pd.to_numeric(dk[salary_col], errors="coerce")

# ---------- Projection baseline ----------
median_salary = dk[salary_col].median()
slope = 3.0 / 1000
a = 55 - slope * median_salary

dk["salary_proj"] = a + slope * dk[salary_col]

if avg_col:
    dk["base_proj"] = 0.7 * dk[avg_col] + 0.3 * dk["salary_proj"]
else:
    dk["base_proj"] = dk["salary_proj"]

# ---------- DataGolf merge ----------
cut_available = False

if dg_file:
    dg = pd.read_csv(dg_file)
    dg = normalize_cols(dg)

    dg_name = find_col(dg, ["player_name", "name", "player"])
    dg["name_key"] = dg[dg_name].map(norm_name)

    col_app = find_col(dg, ["app_true", "sg_app", "app"])
    col_ott = find_col(dg, ["ott_true", "sg_ott", "ott"])
    col_arg = find_col(dg, ["arg_true", "sg_arg", "arg"])
    col_putt = find_col(dg, ["putt_true", "sg_putt", "putt"])
    col_t2g = find_col(dg, ["t2g_true", "sg_t2g", "t2g"])
    col_total = find_col(dg, ["total_true", "sg_total", "total"])

    keep = ["name_key", col_app, col_ott, col_arg, col_putt, col_t2g, col_total]
    keep = [c for c in keep if c]

    dg_small = dg[keep].copy()
    dk = dk.merge(dg_small, on="name_key", how="left")

    if all(c in dk.columns for c in [col_app, col_ott, col_t2g, col_total]):
        z_app = zscore(dk[col_app])
        z_ott = zscore(dk[col_ott])
        z_t2g = zscore(dk[col_t2g])
        z_total = zscore(dk[col_total])

        z_arg = zscore(dk[col_arg]) if col_arg else 0
        z_putt = zscore(dk[col_putt]) if col_putt else 0

        ball_strike = 0.55*z_app + 0.35*z_ott + 0.10*z_arg
        cut_z = 0.7*z_t2g + 0.3*z_total

        penalty = np.where((z_putt > 0.8) & (ball_strike < -0.4), 0.25, 0)
        dk["cut_safety_z"] = cut_z - penalty

        cs = dk["cut_safety_z"]
        dk["cut_safety"] = 100*(cs - cs.min())/(cs.max()-cs.min()+1e-9)

        cut_available = True

if not cut_available:
    dk["cut_safety"] = 0

# ---------- Final SE projection ----------
dk["proj_z"] = zscore(dk["base_proj"])
dk["cut_z2"] = zscore(dk["cut_safety"])

dk["se_proj"] = dk["base_proj"] * (
    1 + 0.06 * (0.75*dk["proj_z"] + 0.25*dk["cut_z2"]).clip(-2,2)/2
)

st.dataframe(
    dk[[name_col, salary_col, "se_proj", "cut_safety"]]
    .sort_values("se_proj", ascending=False)
    .head(40),
    use_container_width=True
)

# ---------- Optimizer ----------
st.subheader("Build Lineup")

if st.button("Generate Lineup"):
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

    pool = dk.copy()

    # Remove players with missing projections
    pool = pool.replace([np.inf, -np.inf], np.nan)
    pool = pool.dropna(subset=["se_proj", salary_col])

    pool = pool.reset_index(drop=True)


    prob = LpProblem("DK_Golf", LpMaximize)
    x = [LpVariable(f"x{i}", 0, 1, LpBinary) for i in range(len(pool))]

    prob += lpSum(pool.loc[i,"se_proj"]*x[i] for i in range(len(pool)))
    prob += lpSum(x) == ROSTER_SIZE
    prob += lpSum(pool.loc[i,salary_col]*x[i] for i in range(len(pool))) <= SALARY_CAP

    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    chosen = [i for i in range(len(pool)) if x[i].value() == 1]
    lineup = pool.loc[chosen]

    st.metric("Salary Used", f"${int(lineup[salary_col].sum()):,}")
    st.metric("Projected Points", f"{lineup['se_proj'].sum():.2f}")

    st.dataframe(
        lineup[[name_col, salary_col, "se_proj", "cut_safety"]],
        use_container_width=True
    )
