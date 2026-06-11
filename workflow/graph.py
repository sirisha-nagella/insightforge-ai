from langgraph.graph import StateGraph

from workflow.state import AgentState

from agents.planner_agent import planner_agent
from agents.data_quality_agent import data_quality_agent
from agents.eda_agent import eda_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.report_agent import report_agent
from agents.model_agent import model_agent

def create_graph():

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("quality", data_quality_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("report", report_agent)
    graph.add_node("model", model_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "quality")
    graph.add_edge("quality", "eda")
    graph.add_edge("eda", "feature_engineering")
    graph.add_edge("feature_engineering", "model")
    graph.add_edge("model", "report")

    graph.set_finish_point("report")

    return graph.compile()