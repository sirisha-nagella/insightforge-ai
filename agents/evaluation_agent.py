from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluation_agent(state):

    model_report = state["model_report"]

    problem_type = state["problem_type"]

    y_test = model_report["y_test"]

    predictions = model_report["predictions"]

    if problem_type == "classification":

        evaluation = {
            "accuracy":
                accuracy_score(y_test, predictions),

            "precision":
                precision_score(y_test, predictions, average="weighted", zero_division=0),

            "recall":
                recall_score(y_test, predictions, average="weighted", zero_division=0),

            "f1":
                f1_score(y_test, predictions, average="weighted", zero_division=0),

            "confusion_matrix":
                confusion_matrix(y_test, predictions).tolist(),
        }

    else:

        evaluation = {
            "mae":
                mean_absolute_error(y_test, predictions),

            "rmse":
                mean_squared_error(y_test, predictions) ** 0.5,

            "r2":
                r2_score(y_test, predictions),
        }

    state["evaluation_report"] = evaluation

    return state
