from langgraph.graph import StateGraph

from workflow.state import AgentState

from agents.planner_agent import planner_agent
from agents.data_quality_agent import data_quality_agent
from agents.eda_agent import eda_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.classification_model_agent import model_agent as classification_model_agent
from agents.regression_model_agent import regression_model_agent
from agents.evaluation_agent import evaluation_agent
from agents.report_agent import report_agent
from agents.visualization_agent import visualization_agent
from agents.model_visualization_agent import model_visualization_agent


def route_problem(state):
    # Router: inspect state, return the key for the conditional edge.
    return state["problem_type"]


def create_graph():

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("quality", data_quality_agent)
    graph.add_node("eda", eda_agent)
    graph.add_node("visualization", visualization_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("classification_model", classification_model_agent)
    graph.add_node("regression_model", regression_model_agent)
    graph.add_node("evaluation", evaluation_agent)
    graph.add_node("model_visualization", model_visualization_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "quality")
    graph.add_edge("quality", "eda")
    graph.add_edge("eda", "visualization")
    graph.add_edge("visualization", "feature_engineering")

    # Dynamic routing: pick the model branch based on problem_type.
    graph.add_conditional_edges(
        "feature_engineering",
        route_problem,
        {
            "classification": "classification_model",
            "regression": "regression_model",
        },
    )

    # Both branches rejoin at evaluation -> report.
    graph.add_edge("classification_model", "evaluation")
    graph.add_edge("regression_model", "evaluation")
    graph.add_edge("evaluation", "model_visualization")
    graph.add_edge("model_visualization", "report")

    graph.set_finish_point("report")

    return graph.compile()
