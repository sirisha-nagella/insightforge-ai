
def planner_agent(state):

    df = state["dataframe"]

    target = state["target_column"]

    unique_values = df[target].nunique()

    if unique_values <= 10:
        state["problem_type"] = "classification"

    else:
        state["problem_type"] = "regression"

    return state

