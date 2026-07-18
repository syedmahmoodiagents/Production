from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    research_result: str
    math_result: str


# ---------------------------------------------------------
# Supervisor
# ---------------------------------------------------------

def supervisor_node(state: AgentState):

    last_message = state["messages"][-1].content.lower()

    # First visit
    if state.get("research_result", "") == "" and "latest" in last_message:
        print("Supervisor -> Research")
        return {"next": "research"}

    # Research completed
    elif (
        state.get("research_result", "") != ""
        and state.get("math_result", "") == ""
    ):
        print("Supervisor -> Math")
        return {"next": "math"}

    # Finished
    else:
        print("Supervisor -> END")
        return {"next": "end"}


# ---------------------------------------------------------
# Research Agent
# ---------------------------------------------------------

def research_node(state: AgentState):

    result = "Company X latest market value = 100"

    print("Research Agent Executed")

    return {
        "research_result": result,
        "messages": [
            AIMessage(content=result)
        ]
    }


# ---------------------------------------------------------
# Math Agent
# ---------------------------------------------------------

def math_node(state: AgentState):

    text = state["research_result"]

    value = 100

    answer = value * 1.5

    print("Math Agent Executed")

    return {
        "math_result": str(answer),
        "messages": [
            AIMessage(content=f"Final Answer = {answer}")
        ]
    }


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

def router(state: AgentState):
    return state["next"]


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_node)
builder.add_node("math", math_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    router,
    {
        "research": "research",
        "math": "math",
        "end": END,
    },
)

builder.add_edge("research", "supervisor")
builder.add_edge("math", "supervisor")

graph = builder.compile()


# ---------------------------------------------------------
# Invoke
# ---------------------------------------------------------

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Find the latest data for Company X and multiply it by 1.5"
            )
        ]
    }
)


print("\n------------- Conversation -------------\n")

for msg in result["messages"]:
    print(msg.content)
