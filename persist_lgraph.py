from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph import START,END

from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):

    goal:str
    tool_history:list
    tool_results:list
    completed_steps:list


def search_flights(state):
    print("Searching Flights")
    state["tool_history"].append(
        "flight_search"
    )
    state["tool_results"].append(
        "Flight Found"
    )
    state["completed_steps"].append(
        "Flights"
    )

    return state


def search_hotels(state):

    print("Searching Hotels")

    state["tool_history"].append(
        "hotel_search"
    )

    state["tool_results"].append(
        "Hotel Found"
    )

    state["completed_steps"].append(
        "Hotels"
    )

    return state


def recommendation(state):

    print("Generating Recommendation")

    state["completed_steps"].append(
        "Recommendation"
    )

    return state


# ---------------------------------------
# Graph
# ---------------------------------------

builder=StateGraph(AgentState)

builder.add_node(
    "Flights",
    search_flights
)

builder.add_node(
    "Hotels",
    search_hotels
)

builder.add_node(
    "Recommend",
    recommendation
)

builder.add_edge(
    START,
    "Flights"
)

builder.add_edge(
    "Flights",
    "Hotels"
)

builder.add_edge(
    "Hotels",
    "Recommend"
)

builder.add_edge(
    "Recommend",
    END
)

# ---------------------------------------
# Persistence
# ---------------------------------------

memory=MemorySaver()

graph=builder.compile(
    checkpointer=memory
)

# ---------------------------------------
# Run
# ---------------------------------------

config={
    "configurable":{"thread_id": "trip001"}
}

graph.invoke(
    {
        "goal":"Goa",
        "tool_history":[],
        "tool_results":[],
        "completed_steps":[]
    },

    config=config
)

# ---------------------------------------
# Retrieve Saved State
# ---------------------------------------

snapshot=graph.get_state(config)
print(snapshot.values)