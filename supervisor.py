import os
from dotenv import load_dotenv
load_dotenv()
# os.getenv("HF_TOKEN")

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langchain.agents import create_agent 
from langgraph_supervisor import create_supervisor

model = ChatOllama(model="llama3.2") 

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b

@tool
def fetch_web_data(query: str) -> str:
    """Searches the web for up-to-date data."""
    return f"Latest market trend data for {query}: structural growth observed."


math_worker = create_agent(
    model=model,
    tools=[multiply_numbers],
    name="math_expert",
    system_prompt="You are a math expert. Only solve math problems. Do not do research."
)

research_worker = create_agent(
    model=model,
    tools=[fetch_web_data],
    name="research_expert",
    system_prompt="You are a researcher. Gather facts but never perform mathematical arithmetic."
)

workflow = create_supervisor(
    agents=[math_worker, research_worker],
    model=model,
    prompt=(
        "You are a team supervisor managing a math_expert and a research_expert. "
        "Analyze the user request and delegate tasks to the appropriate expert. "
        "If a request needs both web data and math calculations, route to research_expert first "
        "and then route that output to the math_expert."
    )
)

app = workflow.compile()

query = "Find the latest data for company X and multiply it by 1.5"
result = app.invoke({"messages": [{"role": "user", "content": query}]})

# print(result['messages'])
for msg in result['messages']:
    print(msg.content)
    print("__________")
