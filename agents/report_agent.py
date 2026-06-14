def report_agent(state):

    dq = state["data_quality_report"]
    fe = state["feature_engineering_report"]
    model_report = state["model_report"]
    evaluation = state["evaluation_report"]
    problem_type = state["problem_type"]

    def _format(d):
        if not d:
            return "  (none)"
        return "\n".join(f"  - {k}: {v}" for k, v in d.items())

    report = f"""
Dataset Overview

Rows: {dq['rows']}
Columns: {dq['columns']}
Duplicates: {dq['duplicates']}

Feature Engineering

Numeric columns: {fe['numeric_columns']}
Categorical columns: {fe['categorical_columns']}

Missing value recommendations:
{_format(fe['missing_value_recommendations'])}

Encoding recommendations:
{_format(fe['encoding_recommendations'])}

Scaling recommendations:
{_format(fe['scaling_recommendations'])}

Drop candidates: {fe['drop_candidates']}
"""

    report += f"""

Model Results

Problem Type:
{problem_type}

Best Model:
{model_report['best_model']}
"""

    if problem_type == "classification":

        report += f"""
Accuracy:
{model_report['best_accuracy']}

Evaluation Metrics

Accuracy: {evaluation['accuracy']:.2f}

Precision: {evaluation['precision']:.2f}

Recall: {evaluation['recall']:.2f}

F1 Score: {evaluation['f1']:.2f}
"""

    else:

        report += f"""
R2 Score:
{model_report['best_score']}

Evaluation Metrics

MAE: {evaluation['mae']:.2f}

RMSE: {evaluation['rmse']:.2f}

R2: {evaluation['r2']:.2f}
"""

    state["final_report"] = report

    state["final_report"] += "\n\n## Insights\n" + state.get("insight_report", "")

    return state
