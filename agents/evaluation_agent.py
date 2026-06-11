from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluation_agent(state):

    model_report = state["model_report"]

    y_test = model_report["y_test"]

    predictions = model_report["predictions"]

    evaluation = {
        "accuracy":
            accuracy_score(y_test, predictions),

        "precision":
            precision_score(y_test, predictions),

        "recall":
            recall_score(y_test, predictions),

        "f1":
            f1_score(y_test, predictions),

        "confusion_matrix":
            confusion_matrix(y_test, predictions).tolist(),
    }

    state["evaluation_report"] = evaluation

    return state
