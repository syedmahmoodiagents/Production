import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_core.messages import HumanMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

model = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))


def calculate_multiply(a: int, b: int) -> int:
    """Multiplies two given integers together and returns the product."""
    return a * b
# Bind the tool directly to the model instance
model_with_tools = model.bind_tools([calculate_multiply])


query = "What is 35 multiplied by 12?"

# Step 1: Model flags and configures the tool request
ai_msg = model_with_tools.invoke([HumanMessage(content=query)])

# Access the structured dictionary output
# print(ai_msg.tool_calls)
# Output: [{'name': 'calculate_multiply', 'args': {'a': 35, 'b': 12}, 'id': 'call_123', 'type': 'tool_call'}]
print(ai_msg.content)