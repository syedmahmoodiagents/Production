import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def weather(city: str) -> str:
    """Returns the weather of a city."""
    return f"The weather in {city} is 28°C and Sunny."

TOOLS = {
    "calculator": calculator,
    "weather": weather
}

agent = llm.bind_tools(list(TOOLS.values()))

STATE_FILE = "state_tools.json"

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
        "tool_results": [],
        "metadata": {
            "session_id": "session_001",
            "status": "running",
            "current_step": "waiting"
        }
    }

def serialize_messages(messages):
    serialized = []
    for m in messages:
        if isinstance(m,HumanMessage):
            serialized.append(
                {
                    "role":"human",
                    "content":m.content
                }
            )

        elif isinstance(m,AIMessage):
            serialized.append(
                {
                    "role":"ai",
                    "content":m.content
                }
            )

        elif isinstance(m,SystemMessage):
            serialized.append(
                {
                    "role":"system",
                    "content":m.content
                }
            )

        elif isinstance(m,ToolMessage):
            serialized.append(
                {
                    "role":"tool",
                    "content":m.content
                }
            )

    return serialized

def restore_messages(saved_messages):
    restored=[]
    for m in saved_messages:

        if m["role"]=="human":
            restored.append(
                HumanMessage(m["content"])
            )

        elif m["role"]=="ai":
            restored.append(
                AIMessage(m["content"])
            )

        elif m["role"]=="system":
            restored.append(
                SystemMessage(m["content"])
            )

        elif m["role"]=="tool":
            restored.append(
                ToolMessage(
                    content=m["content"],
                    tool_call_id="restored"
                )
            )

    return restored


state = load_state()
if state:
    print("\nPrevious Session Found")
    messages = restore_messages(state["messages"])

else:
    print("\nStarting New Session")
    state = create_state()
    messages = [
        SystemMessage(
            "You are a helpful AI assistant."
        )
    ]

while True:
    user_input = input("\nUser : ")
    if user_input.lower()=="exit":
        save_state(state)
        break

    messages.append(
        HumanMessage(user_input)
    )

    state["metadata"]["current_step"]="calling_llm"

    ai_message = agent.invoke(messages)
    messages.append(ai_message)

    if ai_message.tool_calls:
        for call in ai_message.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            print(f"\nTool Selected : {tool_name}")
            # Save Tool History
            state["tool_history"].append(
                {
                    "tool_name":tool_name,
                    "arguments":tool_args,
                    "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )

            # Execute Tool

            try:

                tool_output = TOOLS[tool_name].invoke(tool_args)
                print(f"Tool Output : {tool_output}")
                state["tool_results"].append(
                    {
                        "tool_name":tool_name,
                        "output":tool_output,
                        "status":"success"
                    }
                )
            except Exception as e:
                tool_output = str(e)
                state["tool_results"].append(
                    {
                        "tool_name":tool_name,
                        "output":tool_output,
                        "status":"failed"
                    }
                )

            messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=call["id"]
                )
            )

        final_response = agent.invoke(messages)
        messages.append(final_response)
        print("\nAssistant :")
        print(final_response.content)

    else:
        print("\nAssistant :")
        print(ai_message.content)


    state["messages"] = serialize_messages(messages)
    state["metadata"]["current_step"]="completed"

    save_state(state)
    print("\nSession Saved Successfully")