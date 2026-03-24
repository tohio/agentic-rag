"""
tools.py
--------
Defines the tools available to the LangGraph agent using LangChain's
@tool decorator.

Key difference from the previous version:
    Tools are now decorated with @tool from langchain.tools, which
    automatically generates the JSON schema LangGraph uses to bind
    tools to the LLM for structured tool calling (OpenAI function
    calling format under the hood).

Available tools:
    1. hr_policy_retriever  — searches the HR handbook vector store
    2. calculator           — evaluates safe arithmetic expressions
    3. date_calculator      — computes dates relative to today
    4. escalation_router    — routes to HR contact when needed

Tool design principles:
    - Each tool has a clear docstring — LangChain uses the docstring
      as the tool description passed to the LLM, so it must explain
      WHEN to use the tool, not just WHAT it does
    - Each tool returns a string — LangGraph's ToolNode expects
      string returns from tools
    - Tools are bound to the LLM via llm.bind_tools() in agent.py

Usage:
    from src.agents.tools import get_tools
    tools = get_tools(vectorstore)
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Tool factory
# Tools that require the vectorstore are created via a factory
# function that closes over the vectorstore instance, since @tool
# decorated functions cannot easily accept runtime dependencies
# as constructor arguments.
# -------------------------------------------------------------------

def create_hr_retriever_tool(vectorstore: PineconeVectorStore):
    """
    Factory that creates the hr_policy_retriever tool with the
    vectorstore injected via closure.

    Args:
        vectorstore (PineconeVectorStore): Initialized LangChain vector store.

    Returns:
        Callable: LangChain @tool decorated retriever function.
    """
    from src.retrieval.retriever import retrieve_with_metadata

    @tool
    def hr_policy_retriever(query: str) -> str:
        """
        Search the Meridian Capital Group HR handbook for policy information.

        Use this tool when the user asks any question about:
        - PTO, vacation, sick leave, or time off policies
        - Benefits, health insurance, 401(k), or compensation
        - Performance reviews, ratings, promotions, or bonuses
        - Code of conduct, harassment policy, or disciplinary process
        - Parental leave, bereavement, or other leave types
        - RSU vesting, equity, or other compensation components
        - Any other company policy or HR procedure

        Args:
            query: Natural language search query about HR policies.

        Returns:
            Retrieved context from the HR handbook, or an error message.
        """
        logger.info(f"[Tool: hr_policy_retriever] Query: '{query}'")
        try:
            result = retrieve_with_metadata(vectorstore, query)
            if not result["success"]:
                return (
                    "No relevant policy information found for this query. "
                    "The handbook may not cover this topic."
                )
            logger.info(
                f"[Tool: hr_policy_retriever] "
                f"Retrieved {result['num_results']} chunk(s) | "
                f"max score: {result['max_score']}"
            )
            return result["context"]
        except Exception as e:
            logger.error(f"[Tool: hr_policy_retriever] Error: {e}")
            return f"Retrieval failed: {e}"

    return hr_policy_retriever


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe arithmetic expression and return the result.

    Use this tool when the user's question requires a calculation, such as:
    - 'How much is 15% of my $120,000 salary?'
    - 'What is my total PTO if I have 15 days plus 5 carried over?'
    - 'How many days is 16 weeks of parental leave?'

    Only supports arithmetic operators: +, -, *, /, **, %, and parentheses.
    Do NOT pass code, function calls, or imports — only numeric expressions.

    Args:
        expression: A safe arithmetic expression string, e.g. '120000 * 0.15'

    Returns:
        The computed result as a formatted string.
    """
    logger.info(f"[Tool: calculator] Expression: '{expression}'")

    safe_pattern = r'^[\d\s\+\-\*\/\(\)\.\%\*\*]+$'
    if not re.match(safe_pattern, expression.strip()):
        return (
            f"Unsafe expression: '{expression}'. "
            f"Only arithmetic operators (+, -, *, /, **, %) are allowed."
        )

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        rounded = round(float(result), 4)
        formatted = f"{rounded:,}"
        logger.info(f"[Tool: calculator] Result: {rounded}")
        return (
            f"Expression: {expression}\n"
            f"Result: {formatted}"
        )
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Could not evaluate expression: {e}"


@tool
def date_calculator(action: str, days: int = 0) -> str:
    """
    Compute dates relative to today for policy eligibility questions.

    Use this tool when the user asks date-related questions such as:
    - 'When will my 90-day PTO eligibility period end if I started today?'
    - 'What date is 1 year from now for my RSU vesting cliff?'
    - 'How many days until my annual review if it is in 45 days?'

    Supported actions:
        - 'today'          : returns today's date
        - 'add_days'       : adds N days to today (pass days=N)
        - 'subtract_days'  : subtracts N days from today (pass days=N)
        - 'add_months'     : adds N months to today (pass days=N for months)

    Args:
        action: One of 'today', 'add_days', 'subtract_days', 'add_months'
        days: Number of days or months to add/subtract (default: 0)

    Returns:
        Computed date information as a formatted string.
    """
    logger.info(f"[Tool: date_calculator] action={action} | days={days}")

    try:
        today = datetime.today()

        if action == "today":
            return f"Today's date is {today.strftime('%B %d, %Y')} ({today.strftime('%Y-%m-%d')})"

        if action == "add_days":
            result = today + timedelta(days=days)
        elif action == "subtract_days":
            result = today - timedelta(days=days)
        elif action == "add_months":
            result = today + timedelta(days=days * 30)
        else:
            return (
                f"Unknown action: '{action}'. "
                f"Supported: today, add_days, subtract_days, add_months"
            )

        delta = (result - today).days
        return (
            f"Today: {today.strftime('%B %d, %Y')}\n"
            f"Action: {action} ({days} {'months' if action == 'add_months' else 'days'})\n"
            f"Result: {result.strftime('%B %d, %Y')} ({result.strftime('%Y-%m-%d')})\n"
            f"Days from today: {delta}"
        )

    except Exception as e:
        logger.error(f"[Tool: date_calculator] Error: {e}")
        return f"Date calculation failed: {e}"


@tool
def escalation_router(question: str, reason: str) -> str:
    """
    Route a question to HR when it cannot be answered from the handbook.

    Use this tool when:
    - The HR handbook does not contain enough information to answer
    - The question requires personal employee data (salary, reviews, etc.)
    - The question involves a sensitive or case-specific situation
    - Retrieval has been attempted and returned insufficient results

    Args:
        question: The user's original question.
        reason: Why the agent cannot answer from the handbook alone.

    Returns:
        An escalation message with HR contact information.
    """
    logger.info(f"[Tool: escalation_router] Escalating: '{question[:60]}'")

    return (
        f"This question requires direct HR assistance and cannot be fully "
        f"answered from the employee handbook.\n\n"
        f"Reason: {reason}\n\n"
        f"Please contact Meridian Capital Group HR directly:\n"
        f"- Email: hr@meridiancapitalgroup.com\n"
        f"- Phone: 1-800-555-0199 (HR Helpdesk)\n"
        f"- Portal: workday.meridiancapitalgroup.com\n"
        f"- Hours: Monday–Friday, 9:00 AM – 5:00 PM CST"
    )


# -------------------------------------------------------------------
# Tool registry
# -------------------------------------------------------------------

def get_tools(vectorstore: Optional[PineconeVectorStore] = None) -> list:
    """
    Return the list of LangChain tools for the LangGraph agent.

    The hr_policy_retriever tool requires the vectorstore and is
    created via a factory. All other tools are standalone.

    Args:
        vectorstore (Optional[PineconeVectorStore]): Initialized Pinecone
            vector store. Required for hr_policy_retriever.

    Returns:
        list: List of LangChain @tool decorated callables ready to be
            bound to the LLM via llm.bind_tools().
    """
    tools = [calculator, date_calculator, escalation_router]

    if vectorstore is not None:
        hr_tool = create_hr_retriever_tool(vectorstore)
        tools.insert(0, hr_tool)
    else:
        logger.warning(
            "No vectorstore provided — hr_policy_retriever tool unavailable."
        )

    logger.info(f"Tools loaded: {[t.name for t in tools]}")
    return tools


def get_tool_names(tools: list) -> list[str]:
    """Return the names of all loaded tools."""
    return [t.name for t in tools]


def get_tool_descriptions(tools: list) -> str:
    """
    Return a formatted string of all tool names and descriptions.
    Used for display in the Streamlit UI and agent prompts.
    """
    lines = ["Available tools:\n"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description[:120]}")
    return "\n".join(lines)


# -------------------------------------------------------------------
# Quick test
# python -m src.agents.tools
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Tools Test ---\n")

    # Test calculator
    result = calculator.invoke({"expression": "120000 * 0.15"})
    print(f"Calculator:\n{result}\n")

    # Test date calculator
    result = date_calculator.invoke({"action": "add_days", "days": 90})
    print(f"Date Calculator (90 days from today):\n{result}\n")

    # Test escalation router
    result = escalation_router.invoke({
        "question": "What is my current salary?",
        "reason": "Employee-specific data is not available in the handbook.",
    })
    print(f"Escalation Router:\n{result}\n")

    # Print all tool descriptions (without vectorstore)
    tools = get_tools()
    print(get_tool_descriptions(tools))
