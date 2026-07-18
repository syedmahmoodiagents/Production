import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

import json
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool 

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))



# ============================================================
# Trace File
# ============================================================

TRACE_FILE = "trace.json"

# ============================================================
# Load Existing Trace (Resume)
# ============================================================

if os.path.exists(TRACE_FILE):

    with open(TRACE_FILE, "r") as f:
        traces = json.load(f)

else:

    traces = []

# ============================================================
# Generic Logger
# ============================================================

def log_event(
    category,
    event,
    details=None,
    status="SUCCESS"
):

    if details is None:
        details = {}

    trace = {

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "category": category,

        "event": event,

        "status": status,

        "details": details

    }

    traces.append(trace)

    with open(TRACE_FILE, "w") as f:
        json.dump(traces, f, indent=4)

# ============================================================
# Example Agent Execution
# ============================================================

try:

    # --------------------------------------------------------
    # Session Started
    # --------------------------------------------------------

    log_event(
        category="Session",
        event="Agent Started",
        details={
            "session_id":"session001"
        }
    )

    # --------------------------------------------------------
    # User Query
    # --------------------------------------------------------

    user_query = "Plan my Goa vacation"

    log_event(
        category="Session",
        event="User Query",
        details={
            "query":user_query
        }
    )

    # --------------------------------------------------------
    # Planning Started
    # --------------------------------------------------------

    start = time.time()

    log_event(
        category="Planning",
        event="Planning Started"
    )

    plan = [
        "Search Flights",
        "Search Hotels",
        "Generate Recommendation"
    ]

    log_event(
        category="Planning",
        event="Planning Completed",
        details={
            "plan":plan
        }
    )

    # --------------------------------------------------------
    # Tool Call
    # --------------------------------------------------------

    tool_name = "flight_search"

    tool_input = {
        "destination":"Goa"
    }

    log_event(
        category="Tool",
        event="Tool Invoked",
        details={
            "tool":tool_name,
            "input":tool_input
        }
    )

    # Simulated Tool Execution

    tool_output = "Flight Found for $250"

    log_event(
        category="Tool",
        event="Tool Completed",
        details={
            "tool":tool_name,
            "output":tool_output
        }
    )

    # --------------------------------------------------------
    # LLM Invocation
    # --------------------------------------------------------

    log_event(
        category="LLM",
        event="LLM Started"
    )

    response = (
        "I found a flight to Goa for $250."
    )

    log_event(
        category="LLM",
        event="LLM Completed",
        details={
            "response":response
        }
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    log_event(
        category="Persistence",
        event="Checkpoint Saved",
        details={
            "current_step":2,
            "completed_steps":[
                "Search Flights"
            ]
        }
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    end = time.time()

    log_event(
        category="Performance",
        event="Execution Completed",
        details={
            "latency_seconds": round(end-start,2)
        }
    )

    # --------------------------------------------------------
    # Session End
    # --------------------------------------------------------

    log_event(
        category="Session",
        event="Agent Finished"
    )

except Exception as e:

    log_event(
        category="Error",
        event="Execution Failed",
        status="FAILED",
        details={
            "message":str(e)
        }
    )

print("Trace successfully written to trace.json")