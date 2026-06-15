import streamlit as st
import pandas as pd
import utils.tools as tools

from workflow.graph import create_graph

from utils.db import init_db, save_run, get_history

init_db() # creates the table on first run, does nothing after


st.title("InsightForge AI")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    tools.df = df  # let the Q&A tools query the uploaded dataframe

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

    st.info(
        f"Target: **{target_column}**  |  "
        f"Detected problem type: **{result['problem_type']}**"
    )

    tab_data, tab_report, tab_fe, tab_model, tab_viz, tab_insights = st.tabs(
        [
            "Dataset",
            "Report",
            "Feature Engineering",
            "Model & Evaluation",
            "Visualizations",
            "AI Insights",
        ]
    )

    with tab_data:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    with tab_report:
        st.subheader("Generated Report")
        st.write(result["final_report"])

    with tab_fe:
        st.subheader("Feature Engineering Report")
        st.json(result["feature_engineering_report"])

    with tab_model:
        st.subheader("Model Results")
        st.json(
            {
                k: v
                for k, v in result["model_report"].items()
                if k not in ("y_test", "predictions")
            }
        )

        st.subheader("Evaluation Metrics")
        st.json(result["evaluation_report"])

    with tab_viz:
        st.subheader("Visualizations")
        for name, path in result["visualization_report"].items():
            st.image(path, caption=name)

    with tab_insights:
        st.subheader("AI Insights")
        st.write(result["insight_report"])

    # after the pipeline finishes, save the outcome
    mr = result.get("model_report", {})
    score = mr.get("best_accuracy") or mr.get("best_score")
    save_run(
        uploaded_file.name,
        target_column,
        result.get("problem_type"),
        mr.get("best_model"),
        score,
    )

# Past runs (shown whether or not a file was uploaded this session)
st.subheader("Past Runs")
history = get_history()
if history:
    st.dataframe(
        pd.DataFrame(
            history,
            columns=["Dataset", "Target", "Type", "Best Model", "Score", "When"],
        )
    )
else:
    st.write("No runs yet.")
    

# add a question in streamlit: tools addition

question = st.text_input("Ask a question about your data or past runs")
if question:
    st.write(tools.ask(question))



