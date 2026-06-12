import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def model_visualization_agent(state):

    model_report = state["model_report"]

    visualizations = state.get("visualization_report", {})

    # Feature importance is only present when a tree model (Random Forest) won.

    importances = model_report.get("feature_importances")

    if importances:

        feat_imp = pd.Series(importances).sort_values()

        # Keep the chart readable on wide datasets: show the top 15 features.

        top_n = min(15, len(feat_imp))

        feat_imp = feat_imp.tail(top_n)

        plt.figure(figsize=(8, 6))

        feat_imp.plot.barh()

        plt.title(f"Feature Importance (Top {top_n})")

        plt.tight_layout()

        path = "reports/visualizations/feature_importance.png"

        plt.savefig(path)

        plt.close()

        visualizations["feature_importance"] = path

    state["visualization_report"] = visualizations

    return state
