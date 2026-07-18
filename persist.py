import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


@tool
def weather(city: str) -> str:
    """Return weather information."""
    return f"The weather in {city} is 28°C and Sunny."


TOOLS = {
    "calculator": calculator,
    "weather": weather
}

agent = llm.bind_tools(list(TOOLS.values()))

# JSON Persistence

STATE_FILE = "state.json"

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)

    return None

def create_state():

    return {

        "messages": [],
        "tool_history": [],
        "final_response": "",
        "status": "Running"
    }

def restore_messages(saved):
    messages = []

    for m in saved:

        if m["role"] == "system":
            messages.append(SystemMessage(m["content"]))

        elif m["role"] == "human":
            messages.append(HumanMessage(m["content"]))

        elif m["role"] == "ai":
            messages.append(AIMessage(m["content"]))

    return messages


def serialize_messages(messages):
    serialized = []

    for m in messages:
        if isinstance(m, HumanMessage):
            serialized.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            serialized.append({"role": "ai", "content": m.content})
        elif isinstance(m, SystemMessage):
            serialized.append({"role": "system", "content": m.content})

    return serialized


state = load_state()
if state is None:
    state = create_state()
    messages = [SystemMessage("You are a helpful assistant.")]
else:
    print("\nRestoring Previous Session...\n")
    messages = restore_messages(state["messages"])

while True:
    query = input("\nUser : ")
    if query.lower() == "exit":
        break

    messages.append(HumanMessage(query))
    response = agent.invoke(messages)
    messages.append(response)

    print("\nAssistant :")
    print(response.content)


    if hasattr(response, "tool_calls"):
        for call in response.tool_calls:
            state["tool_history"].append({"tool": call["name"], "arguments": call["args"]})

    state["messages"] = serialize_messages(messages)
    state["final_response"] = response.content

    save_state(state)
    print("\nSession Saved.")

