import streamlit as st
import pandas as pd

from workflow.graph import create_graph

st.title("InsightForge AI")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    workflow = create_graph()

    result = workflow.invoke(
        {
            "dataframe": df,
            "target_column": target_column
        }
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Generated Report")
    st.write(result["final_report"])

    st.subheader(
        "Feature Engineering Report"
    )

    st.json(
        result["feature_engineering_report"]
    )

    st.subheader("Model Results")

    st.json(
        {
            k: v
            for k, v in result["model_report"].items()
            if k not in ("y_test", "predictions")
        }
    )

    st.subheader(
        "Evaluation Metrics"
    )

    st.json(
        result["evaluation_report"]
    )