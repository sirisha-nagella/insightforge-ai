from typing import TypedDict
import pandas as pd

class AgentState(TypedDict):

    dataframe: pd.DataFrame

    target_column: str

    problem_type: str

    data_quality_report: dict

    eda_report: dict

    visualization_report: dict

    feature_engineering_report: dict

    model_report: dict

    evaluation_report: dict

    insight_report: str

    final_report: str
    