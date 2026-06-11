from typing import TypedDict
import pandas as pd

class AgentState(TypedDict):

    dataframe: pd.DataFrame

    data_quality_report: dict

    eda_report: dict

    feature_engineering_report: dict

    model_report: dict

    final_report: str
    