def report_agent(state):

    dq = state["data_quality_report"]

    report = f"""
Dataset Overview

Rows: {dq['rows']}
Columns: {dq['columns']}
Duplicates: {dq['duplicates']}
"""
    
    state["final_report"] = report

    return state

