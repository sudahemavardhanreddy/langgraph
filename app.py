import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

llm_flash = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", google_api_key=api_key)
llm = llm_flash


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
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
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ==========================================
# 4. GRAPH NODES
# ==========================================
def task_input_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- NEW TASK INITIALIZATION ---")
    user_task = input("Enter the coding task (or type 'exit' to quit): ").strip()

    if user_task.lower() == "exit":
        return {"next_step": "exit"}

    return {
        "messages": [HumanMessage(content=user_task)],
        "next_step": "developer",
    }


def real_time_developer(state: CrewState):
    print("\n[Developer] Writing dynamic code using LLM...")
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content
    if isinstance(content, list):
        code_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        code_str = str(content)

    print(code_str)
    return {"code": code_str}


def real_time_tester(state: CrewState):
    print("\n[Tester] Generating dynamic tests and executing code...")
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    content = test_cases
    if isinstance(content, list):
        cases_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        cases_str = str(content)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = f"### EXECUTION OUTPUT:\n{execution_result}\n\n### TEST SCENARIOS EVALUATED:\n{cases_str}"
    return {"report": report}


def manager_decision_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : TEST REPORT ---")
    print(state.get("report", "No report available."))
    print("=" * 50)

    user_input = input("\nCommand (store / another): ").lower().strip()

    if user_input == "store":
        return {"next_step": "archiver"}
    else:
        return {"next_step": "task_input"}


def archiver_node(state: CrewState):
    print("\n[Archiver] Task stored successfully. Closing workflow.")
    return {"next_step": "exit"}


# ==========================================
# 5. GRAPH CONSTRUCTION & ROUTING
# ==========================================
rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")


def route_from_input(state):
    if state.get("next_step") == "exit":
        return END
    return "developer"


rt_workflow.add_conditional_edges("task_input", route_from_input)

rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "manager_decision")


def route_from_decision(state):
    if state.get("next_step") == "archiver":
        return "archiver"
    return "task_input"


rt_workflow.add_conditional_edges("manager_decision", route_from_decision)
rt_workflow.add_edge("archiver", END)

rt_app = rt_workflow.compile()
print("Interactive pipeline compiled and ready for live execution.")


# ==========================================
# 6. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    try:
        rt_app.invoke({"messages": []}, config={"recursion_limit": 50})
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
