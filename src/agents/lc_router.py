"""
router.py
---------
Responsible for classifying incoming queries and routing them to
the appropriate node in the LangGraph state graph.

Key difference from the previous version:
    Uses LangChain's structured output (with_structured_output) instead
    of free-text LLM classification. The LLM is forced to return a
    typed RouteDecision object, eliminating the need to parse and
    validate free-text responses.

    This is the modern LangChain pattern for classification tasks —
    it's more reliable than regex parsing and integrates natively
    with LangGraph's conditional edge routing.

Route categories:
    - hr_retrieval   : policy questions answerable from the handbook
    - calculation    : questions requiring arithmetic
    - date_lookup    : questions requiring date computation
    - multi_step     : complex questions requiring multiple tools
    - escalation     : questions requiring direct HR contact
    - out_of_scope   : questions unrelated to HR or company policy

Usage:
    from src.agents.router import QueryRouter
    router = QueryRouter(llm)
    decision = router.classify(state)
    # Returns RouteDecision(route="hr_retrieval", reasoning="...")
"""

import logging
import os
from typing import Literal, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Typed route decision — LangChain structured output schema
# The LLM is forced to return an object matching this schema,
# eliminating free-text parsing and validation entirely.
# -------------------------------------------------------------------

RouteType = Literal[
    "hr_retrieval",
    "calculation",
    "date_lookup",
    "multi_step",
    "escalation",
    "out_of_scope",
]


class RouteDecision(BaseModel):
    """Structured routing decision returned by the LLM classifier."""

    route: RouteType = Field(
        description=(
            "The route category for this query. Must be one of: "
            "hr_retrieval, calculation, date_lookup, multi_step, "
            "escalation, out_of_scope"
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )


class SubQueryList(BaseModel):
    """Structured decomposition of a multi-step query."""

    sub_queries: list[str] = Field(
        description="List of 2-3 simpler sub-queries decomposed from the original."
    )


# -------------------------------------------------------------------
# Router prompt templates
# -------------------------------------------------------------------

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query router for an HR policy assistant at Meridian Capital Group.\n"
        "Classify the user's query into exactly one route category:\n\n"
        "- hr_retrieval  : Questions about HR policies, benefits, PTO, compensation,\n"
        "                  performance reviews, or code of conduct.\n"
        "- calculation   : Questions requiring arithmetic (salary percentages,\n"
        "                  PTO totals, bonus amounts).\n"
        "- date_lookup   : Questions about dates, eligibility windows, vesting\n"
        "                  schedules, or timelines relative to today.\n"
        "- multi_step    : Complex questions requiring BOTH policy retrieval AND\n"
        "                  calculation or date lookup.\n"
        "- escalation    : Questions requiring personal employee data, case-specific\n"
        "                  situations, or information not in the handbook.\n"
        "- out_of_scope  : Questions completely unrelated to HR or employment.\n\n"
        "{memory_context}"
    )),
    ("human", "{query}"),
])

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query decomposer for an HR policy assistant.\n"
        "Break the complex query into 2-3 simpler sub-queries that can each\n"
        "be answered independently using one of these tools:\n\n"
        "{tool_descriptions}\n\n"
        "Each sub-query should be a complete, self-contained question."
    )),
    ("human", "{query}"),
])


# -------------------------------------------------------------------
# Escalation signal keywords — fast pre-filter before LLM call
# -------------------------------------------------------------------

ESCALATION_SIGNALS = [
    "my salary",
    "my pay",
    "my performance",
    "my bonus",
    "my record",
    "my file",
    "terminated",
    "fired",
    "lawsuit",
    "complaint against",
    "report my manager",
]


class QueryRouter:
    """
    Classifies user queries and routes them to the appropriate
    LangGraph node using LangChain structured output.

    Attributes:
        llm: Base LangChain chat model.
        classifier: LLM chain with structured output bound to RouteDecision.
        decomposer: LLM chain with structured output bound to SubQueryList.
        fallback_route (str): Route to use if classification fails.
    """

    def __init__(self, llm, fallback_route: str = "hr_retrieval"):
        self.llm = llm
        self.fallback_route = fallback_route

        # Bind structured output schemas to the LLM
        # with_structured_output forces the LLM to return typed objects
        self.classifier = (
            ROUTER_PROMPT
            | llm.with_structured_output(RouteDecision)
        )
        self.decomposer = (
            DECOMPOSITION_PROMPT
            | llm.with_structured_output(SubQueryList)
        )

        logger.info(
            f"QueryRouter initialized | "
            f"fallback_route: {fallback_route}"
        )

    def classify(
        self,
        query: str,
        memory_context: Optional[str] = None,
    ) -> RouteDecision:
        """
        Classify a query into a route category using structured output.

        Args:
            query (str): The user's natural language query.
            memory_context (Optional[str]): Formatted conversation history
                from memory.py, for context-aware routing.

        Returns:
            RouteDecision: Typed route decision with route, reasoning,
                and confidence score.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        memory_section = ""
        if memory_context:
            memory_section = (
                f"Conversation context:\n{memory_context}\n\n"
            )

        logger.info(f"Classifying query: '{query[:60]}'")

        try:
            decision: RouteDecision = self.classifier.invoke({
                "query": query.strip(),
                "memory_context": memory_section,
            })

            logger.info(
                f"Route: '{decision.route}' | "
                f"confidence: {decision.confidence:.2f} | "
                f"reasoning: {decision.reasoning[:80]}"
            )
            return decision

        except Exception as e:
            logger.error(f"Router classification failed: {e}. "
                         f"Falling back to '{self.fallback_route}'")
            return RouteDecision(
                route=self.fallback_route,
                reasoning=f"Classification failed ({e}), using fallback route.",
                confidence=0.0,
            )

    def decompose(
        self,
        query: str,
        tool_descriptions: str,
    ) -> list[str]:
        """
        Decompose a multi-step query into simpler sub-queries.

        Uses structured output to get a typed SubQueryList, eliminating
        the need to parse numbered lists from free text.

        Args:
            query (str): The complex multi-step query.
            tool_descriptions (str): Available tool descriptions.

        Returns:
            list[str]: List of sub-queries. Falls back to [query] if
                decomposition returns fewer than 2 sub-queries.
        """
        logger.info(f"Decomposing: '{query[:60]}'")

        try:
            result: SubQueryList = self.decomposer.invoke({
                "query": query.strip(),
                "tool_descriptions": tool_descriptions,
            })

            sub_queries = result.sub_queries

            if len(sub_queries) < 2:
                logger.warning(
                    "Decomposition returned fewer than 2 sub-queries. "
                    "Using original query."
                )
                return [query]

            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            for i, sq in enumerate(sub_queries, 1):
                logger.info(f"  Sub-query {i}: {sq}")

            return sub_queries

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return [query]

    def should_escalate(self, query: str) -> bool:
        """
        Fast pre-filter for obvious escalation signals.

        Checks for personal data requests before making an LLM
        classification call to save latency and cost.

        Args:
            query (str): The user's query.

        Returns:
            bool: True if the query likely needs escalation.
        """
        query_lower = query.lower()
        for signal in ESCALATION_SIGNALS:
            if signal in query_lower:
                logger.info(
                    f"Escalation signal detected: '{signal}'"
                )
                return True
        return False


# -------------------------------------------------------------------
# Quick test
# python -m src.agents.router
# -------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.generation.generator import get_llm
    load_dotenv()

    llm = get_llm()
    router = QueryRouter(llm)

    test_queries = [
        "How many PTO days do I get in my first year?",
        "What is 15% of my $120,000 salary?",
        "When will my 90-day PTO eligibility end if I started today?",
        "How much will my VP bonus be and when is it paid?",
        "What is my current performance rating?",
        "What is the weather today?",
    ]

    print("\n--- Router Test ---\n")
    for query in test_queries:
        decision = router.classify(query)
        print(f"Query      : {query}")
        print(f"Route      : {decision.route}")
        print(f"Confidence : {decision.confidence:.2f}")
        print(f"Reasoning  : {decision.reasoning}")
        print()

    # Test decomposition
    print("--- Decomposition Test ---\n")
    complex_query = "How much will my VP bonus be and when will it be paid out?"
    from src.agents.tools import get_tools, get_tool_descriptions
    tools = get_tools()
    sub_queries = router.decompose(complex_query, get_tool_descriptions(tools))
    print(f"Original  : {complex_query}")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  Sub {i}  : {sq}")
