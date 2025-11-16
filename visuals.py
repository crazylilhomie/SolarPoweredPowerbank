import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fig(fig):
    st.pyplot(fig)

def render_dashboard(df):
    st.subheader("📈 Top 10 Business-Centric Visual Insights")

    numeric = df.select_dtypes(include="number").columns

    figs = []

    if len(numeric) >= 1:
        fig, ax = plt.subplots()
        sns.histplot(df[numeric[0]], kde=True, ax=ax)
        ax.set_title(f"Distribution of {numeric[0]}")
        figs.append(fig)

    if len(numeric) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[numeric[0]], y=df[numeric[1]], ax=ax)
        ax.set_title(f"{numeric[0]} vs {numeric[1]}")
        figs.append(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    figs.append(fig)

    for fig in figs:
        plot_fig(fig)
