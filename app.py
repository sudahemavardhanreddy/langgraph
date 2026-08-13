import os
import io
import sys
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

llm = llm_flash


class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]
    command: Optional[str]


@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return standard output or an error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 specific test scenarios for a coding task."""

    prompt = (
        "You are a Senior QA Engineer. Generate 3 to 5 highly specific "
        "test scenarios for the following coding task:\n\n"
        f"{task_description}\n\n"
        "Include standard cases and edge cases. Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


def task_input_node(state: CrewState):
    if not state.get("messages"):
        return {"next_step": "exit"}

    return {"next_step": "developer"}


def real_time_developer(state: CrewState):
    task = state["messages"][-1].content

    prompt = (
        "Write a clean Python script to solve this coding task. "
        "Only return executable Python code, with no explanation and "
        "no markdown code fences.\n\n"
        f"Task: {task}"
    )

    # Corrected: use llm_flash, which is the initialized LLM variable.
    response = llm_flash.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        code_str = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    else:
        code_str = str(content)

    return {"code": code_str}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):
        cases_str = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in test_cases
        )
    else:
        cases_str = str(test_cases)

    execution_result = run_python_code.invoke(
        {"code": state.get("code", "")}
    )

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )

    return {"report": report}


def manager_decision_node(state: CrewState):
    command = state.get("command", "store").lower().strip()

    return {
        "next_step": "archiver" if command == "store" else "exit"
    }


def archiver_node(state: CrewState):
    return {"next_step": "exit"}


workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)
workflow.add_node("manager_decision", manager_decision_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")


def route_from_input(state: CrewState):
    return END if state.get("next_step") == "exit" else "developer"


workflow.add_conditional_edges(
    "task_input",
    route_from_input,
)

workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager_decision")


def route_from_decision(state: CrewState):
    return (
        "archiver"
        if state.get("next_step") == "archiver"
        else END
    )


workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision,
)

workflow.add_edge("archiver", END)

rt_app = workflow.compile()


app = FastAPI(
    title="Agentic AI Developer & Tester",
    version="1.0.0",
    description="LangGraph developer, tester and reporting agent.",
)


class AgentRequest(BaseModel):
    task: str = Field(
        ...,
        description="Coding task for the agent",
    )
    command: str = Field(
        "store",
        description="store or another",
    )


class AgentResponse(BaseModel):
    task: str
    generated_code: str
    report: str
    next_step: str


@app.get("/")
def root():
    return {
        "message": "Agentic AI Developer & Tester is running.",
        "docs": "/docs",
        "endpoint": "/run",
    }


@app.post("/run", response_model=AgentResponse)
def run_agent(request: AgentRequest):
    command = request.command.lower().strip()

    if command not in {"store", "another"}:
        raise HTTPException(
            status_code=400,
            detail="command must be either 'store' or 'another'",
        )

    state: CrewState = {
        "messages": [HumanMessage(content=request.task)],
        "next_step": "developer",
        "command": command,
    }

    try:
        result = rt_app.invoke(
            state,
            config={"recursion_limit": 50},
        )

        return AgentResponse(
            task=request.task,
            generated_code=result.get("code", ""),
            report=result.get("report", ""),
            next_step=result.get("next_step", "exit"),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
