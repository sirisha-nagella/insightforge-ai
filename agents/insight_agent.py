import json
from utils.llm import ask

def insight_agent(state):
    #only the reports we actually need for insights

    model_report = state.get("model_report", {})

    # drop the large per-row arrays; keep scores / best model / importances
    model_summary = {
        k: v
        for k, v in model_report.items()
        if k not in ("y_test", "predictions")
    }

    context = {
        "problem_type": state.get("problem_type"),
        "target_column": state.get("target_column"),
        "model_report": model_summary,
        "evaluation_report": state.get("evaluation_report"),
    }

    prompt = f"""you are a data scientist explaining results to a business team.
    Based on the results below, write 3-4 short insights in plain English.
    {json.dumps(context, default=str)}"""

    state["insight_report"] = ask(prompt)
    return state

