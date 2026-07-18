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
def flight_search(destination: str) -> str:
    """Search flights"""
    return f"Flight found to {destination} for $250"

@tool
def hotel_search(destination: str) -> str:
    """Search hotels"""
    return f"Hotel booked in {destination} for $120/night"


TOOLS = {
    "flight_search": flight_search,
    "hotel_search": hotel_search
}

agent = llm.bind_tools(list(TOOLS.values()))

STATE_FILE = "state_agent.json"

def save_state(state):

    with open(STATE_FILE,"w") as f:
        json.dump(state,f,indent=4)


def load_state():

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)

    return None


def create_state(goal):

    return {

        "goal": goal,

        "plan":[
            "Search Flights",
            "Search Hotels",
            "Generate Final Recommendation"
        ],

        "current_step":0,
        "completed_steps":[],
        "messages":[],
        "tool_history":[],
        "tool_results":[],
        "memory":{
            "destination":"Goa"
        },
        "status":"Running"

    }

def serialize(messages):
    output=[]
    for m in messages:
        if isinstance(m,SystemMessage):
            output.append({"role":"system", "content":m.content})

        elif isinstance(m,HumanMessage):
            output.append(
                {
                    "role":"human",
                    "content":m.content
                }
            )

        elif isinstance(m,AIMessage):
            output.append(
                {
                    "role":"ai",
                    "content":m.content
                }
            )

        elif isinstance(m,ToolMessage):
            output.append(
                {
                    "role":"tool",
                    "content":m.content
                }
            )

    return output


def restore(saved):
    messages=[]

    for m in saved:
        if m["role"]=="system":
            messages.append(SystemMessage(m["content"]))

        elif m["role"]=="human":
            messages.append(HumanMessage(m["content"]))

        elif m["role"]=="ai":
            messages.append(AIMessage(m["content"]))

        elif m["role"]=="tool":
            messages.append(
                ToolMessage(
                    content=m["content"],
                    tool_call_id="restored"
                )
            )

    return messages


state = load_state()
if state:
    print("\nPrevious Agent State Found")
    print("Resuming From Step:",state["current_step"]+1)
    messages = restore(state["messages"])

else:
    goal = input("Goal : ")
    state = create_state(goal)
    messages = [
        SystemMessage("You are a travel planning agent."),
        HumanMessage(goal)
    ]


while state["current_step"] < len(state["plan"]):

    step = state["plan"][state["current_step"]]

    print("\n==============================")
    print("Executing :",step)
    print("==============================")

    destination = state["memory"]["destination"]

    # -----------------------------------
    # STEP 1
    # -----------------------------------

    if step=="Search Flights":
        tool_name="flight_search"
        tool_output = TOOLS[tool_name].invoke(
            {
                "destination":destination
            }
        )

    elif step=="Search Hotels":
        tool_name="hotel_search"
        tool_output = TOOLS[tool_name].invoke(
            {
                "destination":destination
            }
        )

    else:
        prompt=f"""
        Goal

        {state['goal']}

        Flight

        {state['tool_results'][0]['output']}

        Hotel

        {state['tool_results'][1]['output']}

        Generate final recommendation.
        """

        response=agent.invoke(
            messages+[HumanMessage(prompt)]
        )

        tool_name="LLM"
        tool_output=response.content


    state["tool_history"].append(

        {
            "step":step,
            "tool":tool_name,
            "timestamp":datetime.now().strftime("%H:%M:%S")
        }

    )

    state["tool_results"].append(
        {
            "step":step,
            "tool":tool_name,
            "output":tool_output,
            "status":"success"
        }

    )

    messages.append(
        AIMessage(tool_output)
    )

    state["messages"]=serialize(messages)
    state["completed_steps"].append(step)
    state["current_step"]+=1
    save_state(state)
    print("Checkpoint Saved")

    # -----------------------------------
    # Simulate Crash
    # -----------------------------------

    if state["current_step"]==2:
        raise Exception(
            "Application Crashed !!"
        )

print("\nAgent Finished Successfully")
state["status"]="Completed"
save_state(state)