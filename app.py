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

    workflow = create_graph()

    result = workflow.invoke(
        {
            "dataframe": df
        }
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Generated Report")
    st.write(result["final_report"])
    