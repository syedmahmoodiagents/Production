import uid
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")
from langchain_ollama import ChatOllama

import json
# Added ToolMessage to properly format tool outputs for LangChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

# llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))
# llm = ChatOllama(model="gpt-oss")
llm = ChatOllama(model="llama3.2")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Return weather information."""
    return f"The weather in {city} is 28°C and Sunny."


TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather
}

# Bind tools to the model
agent = llm.bind_tools(list(TOOLS.values()))

# --- THE "HOPE" LOOP IMPLEMENTATION ---

# Start the conversation state with the user prompt
messages = [HumanMessage(content="What is the weather in Tokyo? Also, what is 45 * 12?")]

max_iterations = 5  # Emergency stop to prevent infinite API calls
final_response = None

print("...Starting Agentic Loop...")

for iteration in range(max_iterations):
    print(f"\n--- Iteration {iteration + 1} ---")
    
    
    response = agent.invoke(messages)
    messages.append(response)  # Track the model's response in history

    # Check for the Loop Stop Condition: Did it provide an answer instead of asking for tools?
    if not response.tool_calls:
        print("...Success! The LLM provided a final answer.")
        final_response = response
        break

    # Process all tool calls requested by the model in this turn
    print(f"...LLM requested {len(response.tool_calls)} tool action(s).")
    
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"...Executing '{tool_name}' with arguments: {tool_args}")

        try:
            # Look up the tool dynamically using your TOOLS dict
            selected_tool = TOOLS[tool_name]
            
            # Execute the tool using .invoke() to bypass the 'StructuredTool' error
            tool_output = selected_tool.invoke(tool_args)
            
            # Create a proper ToolMessage containing the tool result
            tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_id)
            messages.append(tool_message)
            print("   ✅ Tool completed successfully.")

        except Exception as tool_error:
            # THE "HOPE" RECOVERY STEP:
            # Instead of crashing, capture the error and pass it right back to the model!
            error_feedback = f"Error: The tool '{tool_name}' failed to execute. Details: {str(tool_error)}. Please adjust your input arguments or use a different tool."
            
            tool_message = ToolMessage(content=error_feedback, tool_call_id=tool_id)
            messages.append(tool_message)
            print(f"...Tool failed. Sending error feedback back to the LLM.")

else:
    print("\n ...Stopped: Reached maximum iterations without a definitive answer.")


print("\n================ FINAL OUTPUT ================")
if final_response:
    print(final_response.content)
else:
    print("No final text response generated.")
