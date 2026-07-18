from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_ollama import ChatOllama


# --------------------------------------------------------
# LLM
# --------------------------------------------------------

model = ChatOllama(model="llama3.2")


# --------------------------------------------------------
# State
# --------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    research_result: str
    math_result: str


# --------------------------------------------------------
# Supervisor Node
# --------------------------------------------------------

def supervisor_node(state: AgentState):

    messages = [
        SystemMessage(
            content="""
You are a supervisor.

Available workers:
1. research
2. math

Rules:
- If the user asks for latest information, facts, web data,
  trends or research -> reply ONLY with:

research

- If research_result already exists and math_result is empty,
  reply ONLY with:

math

- If math_result already exists,
  reply ONLY with:

end

Respond using exactly one word:
research
math
end
"""
        )
    ]

    messages.extend(state["messages"])

    if state.get("research_result"):
        messages.append(
            HumanMessage(
                content=f"Research Result: {state['research_result']}"
            )
        )

    if state.get("math_result"):
        messages.append(
            HumanMessage(
                content=f"Math Result: {state['math_result']}"
            )
        )

    response = model.invoke(messages)

    decision = response.content.lower()

    print(f"\nSupervisor Decision -> {decision}")

    if "research" in decision:
        nxt = "research"

    elif "math" in decision:
        nxt = "math"

    else:
        nxt = "end"

    return {
        "messages": [AIMessage(content=f"Supervisor chose: {decision}")],
        "next": nxt,
    }


# --------------------------------------------------------
# Research Node
# --------------------------------------------------------

def research_node(state: AgentState):

    print("\nResearch Agent Running...")

    response = model.invoke(
        [
            SystemMessage(
                content="""
You are a research expert.

Collect facts only.

Do not perform calculations.
"""
            ),
            state["messages"][0],
        ]
    )

    return {
        "research_result": response.content,
        "messages": [
            AIMessage(content=f"Research:\n{response.content}")
        ],
    }


# --------------------------------------------------------
# Math Node
# --------------------------------------------------------

def math_node(state: AgentState):

    print("\nMath Agent Running...")

    response = model.invoke(
        [
            SystemMessage(
                content="""
You are a mathematics expert.

Use the research information below.

Perform all calculations.

Return the final answer.
"""
            ),
            HumanMessage(content=state["research_result"]),
        ]
    )

    return {
        "math_result": response.content,
        "messages": [
            AIMessage(content=f"Math:\n{response.content}")
        ],
    }


# --------------------------------------------------------
# Router
# --------------------------------------------------------

def router(state: AgentState):
    return state["next"]


# --------------------------------------------------------
# Build Graph
# --------------------------------------------------------

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


# --------------------------------------------------------
# Invoke
# --------------------------------------------------------

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Find the latest data for Company X and multiply it by 1.5"
            )
        ],
        "research_result": "",
        "math_result": "",
        "next": "",
    }
)


# --------------------------------------------------------
# Print Conversation
# --------------------------------------------------------

print("\n==============================")
print("Conversation")
print("==============================\n")

for msg in result["messages"]:
    print(msg.content)
    print("---------------------------")

print("\nResearch Result:")
print(result["research_result"])

print("\nMath Result:")
print(result["math_result"])