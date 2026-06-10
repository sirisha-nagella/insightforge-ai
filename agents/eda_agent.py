def eda_agent(state):

    df = state["dataframe"]

    summary = df.describe(include="all")

    state["eda_report"] = {
        "summary": summary.to_dict()
    }

    return state
