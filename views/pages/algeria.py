import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "algeria_genus_pairs_with_stress_profiles.csv")
PRIMARY_STRESS = ["DTI_H", "SSI_H", "FSI_H"]


# ------------- DATA LOADING ------------- #

@st.cache_data
def load_hybrids(path=INPUT_CSV):
    df = pd.read_csv(path)

    needed = {"Genus_A", "Genus_B", "CSTI", "DTI_H", "SSI_H", "FSI_H", "best_zone_stress"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Missing required columns in {path}: {missing}")
        st.stop()

    if "label" not in df.columns:
        df["label"] = np.nan

    df["pair_name"] = df["Genus_A"] + " × " + df["Genus_B"]
    return df


def normalize_globally(df, cols):
    """Add *_norm columns for given cols using global min/max."""
    s = df.copy()
    for t in cols:
        tmin, tmax = s[t].min(), s[t].max()
        if tmax > tmin:
            s[t + "_norm"] = (s[t] - tmin) / (tmax - tmin)
        else:
            s[t + "_norm"] = 0.5
    return s


# ------------- RADAR CHART ------------- #

def make_radar(hybrid_row, traits):
    vals = [float(hybrid_row[t]) for t in traits]
    labels = [t.replace("_H", "") for t in traits]

    vmin = min(vals); vmax = max(vals)
    if vmax > vmin:
        vals_norm = [(v - vmin) / (vmax - vmin) for v in vals]
    else:
        vals_norm = [0.5] * len(vals)

    vals_norm.append(vals_norm[0])
    labels_cycle = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_norm,
        theta=labels_cycle,
        fill='toself',
        name=hybrid_row["pair_name"]
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Trait fingerprint (normalized within this hybrid)"
    )
    return fig


# ------------- USER-DEFINED SCENARIO ------------- #

def apply_user_scenario(df,
                        w_D, w_S, w_F,
                        bonus_coastal, bonus_plateau, bonus_saharan,
                        penalty_frost,
                        use_normalized=True):
    """
    Compute ScenarioScore from user-chosen parameters.

    Base formula (global):
      Score_base = w_D * D + w_S * S + w_F * F

    Zone modifiers:
      + bonus_zone * S_norm   per zone

    Additional frost penalty (global):
      Score -= penalty_frost * F_norm
    """
    s = df.copy()

    # Always have normalized for modifiers
    s = normalize_globally(s, PRIMARY_STRESS)
    Dn = s["DTI_H_norm"]
    Sn = s["SSI_H_norm"]
    Fn = s["FSI_H_norm"]

    if use_normalized:
        D, S, F = Dn, Sn, Fn
    else:
        D, S, F = s["DTI_H"], s["SSI_H"], s["FSI_H"]

    # Base global score from weights
    score = w_D * D + w_S * S + w_F * F

    # Zone-based salinity bonuses
    z = s["best_zone_stress"]
    score += np.where(z == "Coastal",      bonus_coastal * Sn, 0.0)
    score += np.where(z == "High_Plateau", bonus_plateau * Sn, 0.0)
    score += np.where(z == "Saharan",      bonus_saharan * Sn, 0.0)

    # Frost penalty (global, using normalized F)
    score -= penalty_frost * Fn

    s["ScenarioScore"] = score
    s["scenario_name"] = "User-designed scenario"
    return s


# ------------- STREAMLIT APP ------------- #

def main():
    st.set_page_config(page_title="Hybrid Scenario Designer", layout="wide")
    st.title("Hybrid Scenario Designer – Algerian Climate Zones")

    df = load_hybrids(INPUT_CSV)
    df = normalize_globally(df, PRIMARY_STRESS)

    # -------- SIDEBAR: Scenario Designer -------- #
    st.sidebar.header("Design your scenario")

    st.sidebar.markdown("**Global trait weights** (how much each stress matters overall):")
    w_D = st.sidebar.slider("Weight for DTI_H (drought)",  -2.0, 2.0, 0.5, 0.1)
    w_S = st.sidebar.slider("Weight for SSI_H (salinity)", -2.0, 2.0, 0.3, 0.1)
    w_F = st.sidebar.slider("Weight for FSI_H (frost)",    -2.0, 2.0, -0.2, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Zone-specific salinity boosts** (extra importance of salinity in each zone):")
    bonus_coastal  = st.sidebar.slider("Coastal salinity boost",      0.0, 1.0, 0.2, 0.05)
    bonus_plateau  = st.sidebar.slider("High Plateau salinity boost", 0.0, 1.0, 0.1, 0.05)
    bonus_saharan  = st.sidebar.slider("Saharan salinity boost",      0.0, 1.0, 0.15, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Additional frost penalty** (kills frost-sensitive hybrids):")
    penalty_frost = st.sidebar.slider("Global frost penalty", 0.0, 2.0, 0.3, 0.05)

    st.sidebar.markdown("---")
    use_norm = st.sidebar.checkbox("Use normalized traits for base weights", value=True)
    top_n = st.sidebar.slider("Top N hybrids to inspect", 5, 50, 15)
    zone_filter = st.sidebar.selectbox("Zone filter", ["All", "Saharan", "High_Plateau", "Coastal"], index=0)

    # Apply scenario
    df_scen = apply_user_scenario(
        df,
        w_D=w_D, w_S=w_S, w_F=w_F,
        bonus_coastal=bonus_coastal,
        bonus_plateau=bonus_plateau,
        bonus_saharan=bonus_saharan,
        penalty_frost=penalty_frost,
        use_normalized=use_norm,
    )

    if zone_filter != "All":
        df_view = df_scen[df_scen["best_zone_stress"] == zone_filter].copy()
    else:
        df_view = df_scen.copy()

    # -------- ROW 1: Scenario summary -------- #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scenario emphasis on stresses")
        w_df = pd.DataFrame({
            "Trait": ["DTI_H", "SSI_H", "FSI_H"],
            "Weight": [w_D, w_S, w_F],
        })
        fig_w = px.bar(
            w_df,
            x="Trait",
            y="Weight",
            color="Trait",
            color_discrete_sequence=["#1b9e77", "#7570b3", "#d95f02"],
        )
        fig_w.update_layout(showlegend=False, yaxis_title="Global weight (base influence)")
        st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        st.subheader("Zone winners under your scenario")
        zones = ["Saharan", "High_Plateau", "Coastal"]
        scores = []
        for z in zones:
            sub = df_scen[df_scen["best_zone_stress"] == z]
            if sub.empty:
                scores.append(0.0)
            else:
                scores.append(sub["ScenarioScore"].nlargest(top_n).mean())
        z_df = pd.DataFrame({"Zone": zones, "MeanTopScore": scores})
        fig_z = px.bar(
            z_df,
            x="Zone",
            y="MeanTopScore",
            color="Zone",
            color_discrete_sequence=["#e31a1c", "#33a02c", "#1f78b4"],
        )
        fig_z.update_layout(showlegend=False, yaxis_title=f"Mean ScenarioScore (top {top_n})")
        st.plotly_chart(fig_z, use_container_width=True)

    # Explain briefly what user has done
    st.markdown(
        f"""
        **Your scenario summary:**
        - Global weights: DTI_H = {w_D:.2f}, SSI_H = {w_S:.2f}, FSI_H = {w_F:.2f} (negative = penalty).
        - Extra salinity boost: Coastal = {bonus_coastal:.2f}, High Plateau = {bonus_plateau:.2f}, Saharan = {bonus_saharan:.2f}.
        - Global frost penalty: {penalty_frost:.2f} (applied via FSI_H_norm).
        """
    )

    # -------- ROW 2: Trade-off scatter -------- #
    st.subheader("Drought–salinity trade-off under your scenario")

    if len(df_view) > 0:
        fig_sc = px.scatter(
            df_view,
            x="DTI_H",
            y="SSI_H",
            color="ScenarioScore",
            color_continuous_scale="Viridis",
            hover_data=["pair_name", "CSTI", "best_zone_stress", "label"],
        )
        fig_sc.update_layout(
            xaxis_title="DTI_H (drought tolerance)",
            yaxis_title="SSI_H (salinity stress)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("No hybrids in this view for the selected settings.")

    # -------- ROW 3: Top hybrids table + radar -------- #
    st.subheader("Top hybrids for your scenario")

    if len(df_view) > 0:
        ranked = df_view.sort_values("ScenarioScore", ascending=False).head(top_n)
        show_cols = [
            "pair_name", "best_zone_stress", "ScenarioScore",
            "CSTI", "DTI_H", "SSI_H", "FSI_H", "label"
        ]
        show_cols = [c for c in show_cols if c in ranked.columns]

        st.dataframe(ranked[show_cols].reset_index(drop=True), use_container_width=True, height=320)

        idx = st.selectbox(
            "Select a hybrid for detailed fingerprint",
            options=list(range(len(ranked))),
            format_func=lambda i: ranked.iloc[i]["pair_name"],
        )
        row = ranked.iloc[idx]
        radar_fig = make_radar(row, PRIMARY_STRESS)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("No hybrids available for this view.")

if __name__ == "__main__":
    main()
