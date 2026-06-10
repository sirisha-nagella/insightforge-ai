def feature_engineering_agent(state):

    df = state["dataframe"]

    numeric_columns = list(
        df.select_dtypes(include=["number"]).columns
    )

    categorical_columns = list(
        df.select_dtypes(include=["object"]).columns
    )

    # Missing value handling recommendations:

    missing_values = df.isnull().sum().to_dict()

    missing_recommendations = {}

    for col, count in missing_values.items():

        if count > 0:

            if col in numeric_columns:
                missing_recommendations[col] = "Fill with median"
            else:
                missing_recommendations[col] = "Fill with mode"

    # For categorical columns:

    encoding_recommendations = {}

    for col in categorical_columns:

        unique_count = df[col].nunique()

        if unique_count <= 10:

            encoding_recommendations[col] = (
                "One-Hot Encoding"
            )

        else:

            encoding_recommendations[col] = (
                "Label Encoding"
            )

    # For numerical columns:

    scaling_recommendations = {}

    for col in numeric_columns:

        if df[col].nunique() > 20:

            scaling_recommendations[col] = (
                "StandardScaler"
            )

    # Detect ID columns:

    drop_candidates = []

    for col in df.columns:

        if "id" in col.lower():

            drop_candidates.append(col)

    state["feature_engineering_report"] = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_value_recommendations": missing_recommendations,
        "encoding_recommendations": encoding_recommendations,
        "scaling_recommendations": scaling_recommendations,
        "drop_candidates": drop_candidates
    }

    return state
