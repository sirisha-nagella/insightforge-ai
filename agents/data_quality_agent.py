def data_quality_agent(state):

    df = state["dataframe"]

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

    state["data_quality_report"] = report

    return state
