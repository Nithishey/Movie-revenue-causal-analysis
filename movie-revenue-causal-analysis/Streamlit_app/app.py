"""Movie Revenue Driver Analysis - Streamlit dashboard.

Run locally:  streamlit run app.py
Needs:        data/clean_movies.csv and data/causal_results.csv (exported by the Colab notebook)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
EXPLORER_VARS = ["budget", "runtime", "popularity", "cast_size", "crew_size", "n_keywords", "release_year"]


# ---------- data ----------
@st.cache_data
def load_movies() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "clean_movies.csv", parse_dates=["release_date"])


@st.cache_data
def load_causal() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "causal_results.csv")


def genre_names(df: pd.DataFrame) -> list:
    return sorted(c.replace("genre_", "") for c in df.columns if c.startswith("genre_"))


def apply_filters(df, years, genres, budget_known_only, franchise):
    out = df[df["release_year"].between(years[0], years[1])]
    if genres:
        out = out[out[[f"genre_{g}" for g in genres]].sum(axis=1) > 0]
    if budget_known_only:
        out = out[out["budget_missing"] == 0]
    if franchise != "All":
        out = out[out["in_collection"] == int(franchise == "Franchise only")]
    return out


def budget_change_effect(current_m: float, new_m: float, elasticity: float) -> float:
    """Percent change in revenue implied by a log-log elasticity when budget moves current -> new."""
    return ((new_m / current_m) ** elasticity - 1) * 100


# ---------- pages ----------
def page_overview(df):
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(df):,}")
    c2.metric("Median revenue", f"${df['revenue'].median() / 1e6:,.1f}M")
    c3.metric("Median budget (known)", f"${df['budget'].median() / 1e6:,.1f}M")
    c4.metric("Franchise films", f"{df['in_collection'].mean():.0%}")

    left, right = st.columns(2)
    log_df = df.assign(log10_revenue=np.log10(df["revenue"]))   # bin in log space (histogram + log axis renders empty)
    left.plotly_chart(px.histogram(log_df, x="log10_revenue", nbins=40,
                                   labels={"log10_revenue": "log10(revenue in USD): 6 = $1M, 8 = $100M"},
                                   title="Revenue distribution (log10 scale)"),
                      use_container_width=True)
    yearly = df.groupby("release_year", as_index=False)["revenue"].median()
    right.plotly_chart(px.line(yearly, x="release_year", y="revenue", title="Median revenue by release year"),
                       use_container_width=True)

    rows = []
    for g in genre_names(df):
        sub = df[df[f"genre_{g}"] == 1]
        if len(sub):
            rows.append({"genre": g, "median_revenue_M": sub["revenue"].median() / 1e6, "movies": len(sub)})
    if rows:
        gdf = pd.DataFrame(rows).sort_values("median_revenue_M")
        st.plotly_chart(px.bar(gdf, x="median_revenue_M", y="genre", orientation="h", hover_data=["movies"],
                               title="Median revenue by genre (USD millions)"), use_container_width=True)


def page_explorer(df):
    st.header("Explorer")
    xvar = st.selectbox("X variable", EXPLORER_VARS)
    plot_df = df.assign(franchise=df["in_collection"].map({1: "Franchise", 0: "Standalone"}))
    st.plotly_chart(px.scatter(plot_df, x=xvar, y="revenue", color="franchise", log_y=True,
                               log_x=(xvar in ("budget", "popularity")), hover_data=["title", "release_year"],
                               opacity=0.6, title=f"Revenue vs {xvar}",
                               color_discrete_map={"Franchise": "#EF553B", "Standalone": "#636EFA"}), use_container_width=True)
    st.subheader("Top 10 by revenue")
    top = df.nlargest(10, "revenue")[["title", "release_year", "budget", "revenue"]]
    st.dataframe(top, use_container_width=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_movies.csv", mime="text/csv")


def page_causal(causal):
    st.header("Causal results")
    st.write("Estimates from the Colab notebook (log-scale coefficients with 95% confidence intervals). "
             "Adjusting only for confounders in the DAG gives the causal estimate; the WRONG row shows the bias "
             "from adjusting for consequences of budget.")
    question = st.selectbox("Question", causal["question"].unique())
    sub = causal[causal["question"] == question].copy()
    fig = go.Figure(go.Scatter(
        x=sub["coef"], y=sub["model"], mode="markers",
        error_x=dict(type="data", symmetric=False,
                     array=sub["ci_high"] - sub["coef"], arrayminus=sub["coef"] - sub["ci_low"])))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=question, xaxis_title="coefficient (effect on log revenue)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sub[["model", "coef", "ci_low", "ci_high", "n"]], use_container_width=True)
    st.caption("Limitations: unmeasured confounders (star power, marketing), movies with unknown budget excluded, "
               "and sensitivity to very small revenue values.")


def page_whatif(causal):
    st.header("Budget what-if")
    row = causal[(causal["question"] == "Q1 budget") & (causal["model"] == "Adjusted (backdoor set)")].iloc[0]
    c1, c2 = st.columns(2)
    current = c1.number_input("Current budget (USD millions)", min_value=0.1, value=20.0, step=1.0)
    new = c2.number_input("New budget (USD millions)", min_value=0.1, value=30.0, step=1.0)

    est = budget_change_effect(current, new, row["coef"])
    low = budget_change_effect(current, new, row["ci_low"])
    high = budget_change_effect(current, new, row["ci_high"])
    st.metric("Expected revenue change", f"{est:+.1f}%", help="Point estimate from the adjusted log-log model")
    st.write(f"95% interval: **{low:+.1f}% to {high:+.1f}%**  (elasticity {row['coef']:.2f})")
    st.warning("This is an average effect for movies with reported budgets, based on observed data and the assumptions "
               "in the DAG. It is not a guarantee for any single film.")


# ---------- app ----------
def main():
    st.set_page_config(page_title="Movie Revenue Drivers", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Revenue Driver Analysis")
    movies, causal = load_movies(), load_causal()

    page = st.sidebar.radio("Page", ["Overview", "Explorer", "Causal results", "Budget what-if"])
    st.sidebar.header("Filters")
    lo, hi = int(movies["release_year"].min()), int(movies["release_year"].max())
    years = st.sidebar.slider("Release years", lo, hi, (lo, hi))
    genres = st.sidebar.multiselect("Genres (any of)", genre_names(movies))
    known = st.sidebar.checkbox("Only movies with known budget", value=False)
    franchise = st.sidebar.selectbox("Franchise", ["All", "Franchise only", "Standalone only"])

    df = apply_filters(movies, years, genres, known, franchise)
    if df.empty:
        st.warning("No movies match the filters.")
        return
    st.sidebar.caption(f"{len(df):,} movies selected")

    if page == "Overview":
        page_overview(df)
    elif page == "Explorer":
        page_explorer(df)
    elif page == "Causal results":
        page_causal(causal)
    else:
        page_whatif(causal)


if __name__ == "__main__":
    main()
