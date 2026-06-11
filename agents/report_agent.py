def report_agent(state):

    dq = state["data_quality_report"]
    fe = state["feature_engineering_report"]
    model_report = state["model_report"]
    evaluation = state["evaluation_report"]

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

Best Model:
{model_report['best_model']}

Accuracy:
{model_report['best_accuracy']}
"""
    
    report += f"""

Evaluation Metrics

Accuracy: {evaluation['accuracy']:.2f}

Precision: {evaluation['precision']:.2f}

Recall: {evaluation['recall']:.2f}

F1 Score: {evaluation['f1']:.2f}
"""
    
    state["final_report"] = report

    return state

