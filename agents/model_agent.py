from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def model_agent(state):

    df = state["dataframe"]

    target_column = "Survived"

    X = df.drop(columns=[target_column])

    y = df[target_column]

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

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(),

        "Random Forest":
            RandomForestClassifier()
    }

    # Train and compare:

    # Loop

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

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = round(
            accuracy,
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

        "best_accuracy": results[best_model],

        "y_test": y_test.tolist(),

        "predictions": best_predictions.tolist()

    }

    return state
