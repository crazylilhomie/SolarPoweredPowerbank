import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fig(fig):
    st.pyplot(fig)

def render_dashboard(df):
    st.subheader("📈 Top Business-Centric Visual Insights")

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.empty:
        st.warning("No numeric columns found for visualizations.")
        return

    figs = []

    # Plot 1 — Histogram of first numeric column
    col1 = numeric_df.columns[0]
    fig, ax = plt.subplots()
    sns.histplot(numeric_df[col1], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col1}")
    figs.append(fig)

    # Plot 2 — Scatter plot if at least 2 numeric columns exist
    if len(numeric_df.columns) >= 2:
        col2 = numeric_df.columns[1]
        fig, ax = plt.subplots()
        sns.scatterplot(x=numeric_df[col1], y=numeric_df[col2], ax=ax)
        ax.set_title(f"{col1} vs {col2}")
        figs.append(fig)

    # Plot 3 — Correlation heatmap (safe)
    if len(numeric_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        figs.append(fig)
    else:
        st.info("Correlation heatmap requires at least two numeric columns.")

    # Display all figures
    for fig in figs:
        plot_fig(fig)
