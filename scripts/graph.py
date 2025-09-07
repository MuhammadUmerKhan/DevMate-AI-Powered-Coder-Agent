from langchain_core.language_models import BaseChatModel
from langgraph.constants import END
from langgraph.graph import StateGraph

from agents import planner_agent, architect_agent, coder_agent

def create_workflow(llm: BaseChatModel) -> StateGraph:
    """Creates and compiles the LangGraph workflow."""
    graph = StateGraph(dict)

    graph.add_node("planner", lambda state: planner_agent(state, llm))
    graph.add_node("architect", lambda state: architect_agent(state, llm))
    graph.add_node("coder", lambda state: coder_agent(state, llm))

    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "coder")
    graph.add_conditional_edges(
        "coder",
        lambda s: "END" if s.get("status") == "DONE" else "coder",
        {"END": END, "coder": "coder"}
    )

    graph.set_entry_point("planner")
    return graph.compile()