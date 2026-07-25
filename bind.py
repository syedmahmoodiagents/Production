import uid
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")
from langchain_ollama import ChatOllama

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

# llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))
# llm = ChatOllama(model="gpt-oss")
llm = ChatOllama(model="llama3.2")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


@tool
def get_weather(city: str) -> str:
    """Return weather information."""
    return f"The weather in {city} is 28°C and Sunny."


TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather
}

agent = llm.bind_tools(list(TOOLS.values()))

response = agent.invoke("What is the weather in Tokyo?")

# print(response.tool_calls[0]['args'])

if response.tool_calls:
    tool_call = response.tool_calls[0]

    # tool_result = get_weather(tool_call["args"]) 
    # this is cann't be used directly because of @tool decorator

    tool_result = get_weather.invoke(tool_call["args"])

    final_response = agent.invoke([
        {"role": "user", "content": "What is the weather in Tokyo?"},
        response,
        {"role": "tool", "content": str(tool_result), "tool_call_id": tool_call["id"]}
    ])

print(final_response)
