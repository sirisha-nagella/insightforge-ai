from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import(
    mean_absolute_error,
    r2_score
)


def regression_model_agent(state):

    df = state["dataframe"]

    target_column = state["target_column"]

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # Drop rows with a missing target (labels can't be imputed).

    valid = y.notna()

    X = X[valid]

    y = y[valid]

    # Drop feature columns that are entirely missing (no signal, breaks imputation).

    X = X.dropna(axis=1, how="all")

    # Numeric Columns:

    for col in X.select_dtypes(include=["number"]).columns:

        X[col] = X[col].fillna(
            X[col].median()
        )

    # Categorical columns:

    for col in X.select_dtypes(include=["object"]).columns:

        X[col] = X[col].fillna(
            X[col].mode()[0]
        )

    # Encode categorical variables:

    for col in X.select_dtypes(include=["object"]).columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col]
        )

    # Train-Test Split:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train models:

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(random_state=42),

        "Random Forest":
            RandomForestRegressor(random_state=42)
    }

    # Train and compare (best = highest R2):

    results = {}

    model_predictions = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        score = r2_score(
            y_test,
            predictions
        )

        results[name] = round(
            score,
            4
        )

        model_predictions[name] = predictions

    # Determine Best Model

    best_model = max(
        results,
        key=results.get
    )

    best_predictions = model_predictions[best_model]

    # Store Results:

    state["model_report"] = {

        "all_models": results,

        "best_model": best_model,

        "best_score": results[best_model],

        "y_test": y_test.tolist(),

        "predictions": best_predictions.tolist()

    }

    # Feature importances are only available for tree models (Random Forest).

    if best_model == "Random Forest":

        state["model_report"]["feature_importances"] = dict(
            zip(
                X.columns,
                models[best_model].feature_importances_.tolist()
            )
        )

    return state
