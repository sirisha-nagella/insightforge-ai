from langgraph.graph import StateGraph

from workflow.state import AgentState

from agents.planner_agent import planner_agent
from agents.data_quality_agent import data_quality_agent
from agents.eda_agent import eda_agent
from agents.report_agent import report_agent

def create_graph():

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("quality", data_quality_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "quality")
    graph.add_edge("quality", "eda")
    graph.add_edge("eda", "report")

    graph.set_finish_point("report")

    return graph.compile()