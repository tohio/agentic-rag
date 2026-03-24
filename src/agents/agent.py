"""
agent.py
--------
The core LangGraph agent for the agentic RAG pipeline.

Implements the agent as a StateGraph — a directed graph where:
    - Nodes are actions  (route, retrieve, calculate, generate, etc.)
    - Edges are transitions (conditional routing between nodes)
    - State is a typed dict passed between nodes at each step

This is the key architectural difference from rag-pipeline and from
the previous imperative agent loop. LangGraph makes the reasoning
process explicit, inspectable, and debuggable — each node is a
discrete step with clear inputs and outputs.

Graph structure:
    START
      ↓
    [route_query]          — classify query, check escalation signals
      ↓ (conditional)
    [retrieve]             — hr_retrieval route
    [calculate]            — calculation route
    [date_lookup]          — date_lookup route
    [decompose_multi_step] — multi_step route
    [handle_escalation]    — escalation route
    [handle_out_of_scope]  — out_of_scope route
      ↓
    [check_retrieval]      — verify retrieval quality, re-retrieve if poor
      ↓ (conditional)
    [generate]             — synthesize final answer from context
      ↓
    [update_memory]        — persist turn to conversation memory
      ↓
    END

Usage:
    from src.agents.agent import build_graph, AgentState
    graph = build_graph(llm, vectorstore, memory)
    result = graph.invoke({"query": "How many PTO days do I get?"})
"""

import logging
import os
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 2
DEFAULT_MIN_RETRIEVAL_SCORE = 0.35


# -------------------------------------------------------------------
# Agent State
# Typed dict passed between all nodes in the graph.
# Each node reads from state and returns a partial update.
# -------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Fields are updated incrementally — each node returns only the
    fields it modifies, and LangGraph merges them into the state.
    """
    # Input
    query: str

    # Routing
    route: str
    route_reasoning: str
    route_confidence: float

    # Retrieval
    context: str
    retrieved_results: list        # List[Tuple[Document, float]]
    retrieval_score: float
    retrieval_attempts: int

    # Tool outputs (for calculation and date_lookup routes)
    tool_output: str

    # Sub-queries (for multi_step route)
    sub_queries: list[str]
    sub_query_results: list[str]

    # Generation
    raw_answer: str
    answer: str
    reasoning: str

    # Tool call history (for UI display)
    tool_calls: list[dict]

    # Memory
    memory_context: str

    # Final response
    sources: list[dict]
    num_sources: int
    iterations: int
    error: Optional[str]


# -------------------------------------------------------------------
# Node functions
# Each node receives the full state and returns a partial update dict.
# -------------------------------------------------------------------

def route_query(state: AgentState, router) -> dict:
    """
    Node: Classify the query and determine the routing path.

    Uses QueryRouter to classify the query into a route category.
    Also runs the fast escalation pre-filter before the LLM call.

    Args:
        state (AgentState): Current graph state.
        router (QueryRouter): Initialized query router.

    Returns:
        dict: Partial state update with route, reasoning, confidence.
    """
    query = state["query"]
    memory_context = state.get("memory_context", "")

    logger.info(f"[Node: route_query] Query: '{query[:60]}'")

    # Fast escalation pre-filter
    if router.should_escalate(query):
        logger.info("[Node: route_query] Escalation signal detected")
        return {
            "route": "escalation",
            "route_reasoning": "Query contains personal data or sensitive signal.",
            "route_confidence": 1.0,
            "iterations": state.get("iterations", 0) + 1,
        }

    decision = router.classify(query, memory_context)

    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
        "route_confidence": decision.confidence,
        "iterations": state.get("iterations", 0) + 1,
    }


def retrieve(state: AgentState, vectorstore) -> dict:
    """
    Node: Retrieve relevant document chunks from Pinecone.

    Args:
        state (AgentState): Current graph state.
        vectorstore: Initialized LangChain Pinecone vector store.

    Returns:
        dict: Partial state update with context, results, and score.
    """
    from src.retrieval.retriever import retrieve_with_metadata

    query = state["query"]
    logger.info(f"[Node: retrieve] Query: '{query[:60]}'")

    result = retrieve_with_metadata(vectorstore, query)

    tool_call = {
        "tool": "hr_policy_retriever",
        "success": result["success"],
        "output_preview": result["context"][:150] if result["success"] else "No results",
    }

    return {
        "context": result["context"],
        "retrieved_results": result["results"],
        "retrieval_score": result["max_score"],
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "tool_calls": state.get("tool_calls", []) + [tool_call],
    }


def check_retrieval(state: AgentState, router) -> dict:
    """
    Node: Inspect retrieval quality and reformulate if score is low.

    If the retrieval score is below the minimum threshold and we
    haven't exceeded max retries, reformulate the query and re-retrieve.

    Args:
        state (AgentState): Current graph state.
        router: QueryRouter (used for query reformulation via llm).

    Returns:
        dict: Partial state update, potentially with a reformulated query.
    """
    score = state.get("retrieval_score", 0.0)
    attempts = state.get("retrieval_attempts", 0)
    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", DEFAULT_MIN_RETRIEVAL_SCORE))
    max_retries = int(os.getenv("MAX_AGENT_ITERATIONS", DEFAULT_MAX_RETRIES))

    logger.info(
        f"[Node: check_retrieval] Score: {score:.3f} | "
        f"Attempts: {attempts} | Min: {min_score}"
    )

    if score < min_score and attempts <= max_retries:
        # Reformulate the query for retry
        reformulated = _reformulate_query(router.llm, state["query"])
        logger.info(f"[Node: check_retrieval] Reformulated: '{reformulated}'")
        return {"query": reformulated}

    return {}


def calculate(state: AgentState, tools: list) -> dict:
    """
    Node: Execute the calculator tool for arithmetic queries.

    Args:
        state (AgentState): Current graph state.
        tools (list): List of loaded LangChain tools.

    Returns:
        dict: Partial state update with tool output.
    """
    from src.agents.tools import calculator

    query = state["query"]
    logger.info(f"[Node: calculate] Query: '{query[:60]}'")

    # Ask LLM to extract the arithmetic expression from the query
    expression = _extract_expression(state, tools)
    result = calculator.invoke({"expression": expression})

    tool_call = {
        "tool": "calculator",
        "success": "Error" not in result,
        "output_preview": result[:150],
    }

    return {
        "tool_output": result,
        "context": f"Calculation result:\n{result}",
        "tool_calls": state.get("tool_calls", []) + [tool_call],
    }


def date_lookup(state: AgentState, tools: list) -> dict:
    """
    Node: Execute the date_calculator tool for date queries.

    Args:
        state (AgentState): Current graph state.
        tools (list): List of loaded LangChain tools.

    Returns:
        dict: Partial state update with tool output.
    """
    from src.agents.tools import date_calculator

    query = state["query"]
    logger.info(f"[Node: date_lookup] Query: '{query[:60]}'")

    action, days = _extract_date_params(state, tools)
    result = date_calculator.invoke({"action": action, "days": days})

    tool_call = {
        "tool": "date_calculator",
        "success": "Error" not in result and "Unknown" not in result,
        "output_preview": result[:150],
    }

    return {
        "tool_output": result,
        "context": f"Date calculation result:\n{result}",
        "tool_calls": state.get("tool_calls", []) + [tool_call],
    }


def decompose_multi_step(state: AgentState, router, vectorstore) -> dict:
    """
    Node: Handle multi-step queries by decomposing and executing sub-queries.

    Decomposes the query into sub-queries, routes and executes each,
    then accumulates results for synthesis in the generate node.

    Args:
        state (AgentState): Current graph state.
        router (QueryRouter): For decomposition and sub-query routing.
        vectorstore: For retrieval sub-queries.

    Returns:
        dict: Partial state update with combined context from all sub-queries.
    """
    from src.agents.tools import get_tools, get_tool_descriptions
    from src.retrieval.retriever import retrieve_with_metadata
    from src.agents.tools import calculator, date_calculator

    query = state["query"]
    logger.info(f"[Node: decompose_multi_step] Query: '{query[:60]}'")

    tools = get_tools(vectorstore)
    sub_queries = router.decompose(query, get_tool_descriptions(tools))

    accumulated_context = []
    tool_calls = list(state.get("tool_calls", []))

    for i, sub_query in enumerate(sub_queries, 1):
        logger.info(f"  Sub-query {i}: '{sub_query}'")
        decision = router.classify(sub_query)
        sub_route = decision.route

        if sub_route == "hr_retrieval":
            result = retrieve_with_metadata(vectorstore, sub_query)
            if result["success"]:
                accumulated_context.append(
                    f"[Policy retrieval for: '{sub_query}']\n{result['context']}"
                )
                tool_calls.append({
                    "tool": "hr_policy_retriever",
                    "success": True,
                    "output_preview": result["context"][:120],
                })
        elif sub_route == "calculation":
            expression = _extract_expression_from_query(router.llm, sub_query)
            result = calculator.invoke({"expression": expression})
            accumulated_context.append(
                f"[Calculation for: '{sub_query}']\n{result}"
            )
            tool_calls.append({
                "tool": "calculator",
                "success": "Error" not in result,
                "output_preview": result[:120],
            })
        elif sub_route == "date_lookup":
            action, days = _extract_date_params_from_query(router.llm, sub_query)
            result = date_calculator.invoke({"action": action, "days": days})
            accumulated_context.append(
                f"[Date lookup for: '{sub_query}']\n{result}"
            )
            tool_calls.append({
                "tool": "date_calculator",
                "success": "Error" not in result,
                "output_preview": result[:120],
            })

    combined_context = "\n\n---\n\n".join(accumulated_context) if accumulated_context else ""

    return {
        "sub_queries": sub_queries,
        "sub_query_results": accumulated_context,
        "context": combined_context,
        "tool_calls": tool_calls,
    }


def handle_escalation(state: AgentState) -> dict:
    """
    Node: Generate an escalation response with HR contact details.

    Args:
        state (AgentState): Current graph state.

    Returns:
        dict: Partial state update with escalation answer.
    """
    from src.agents.tools import escalation_router

    query = state["query"]
    route_reasoning = state.get("route_reasoning", "Unable to answer from handbook.")
    logger.info(f"[Node: handle_escalation] Escalating: '{query[:60]}'")

    result = escalation_router.invoke({
        "question": query,
        "reason": route_reasoning,
    })

    tool_call = {
        "tool": "escalation_router",
        "success": True,
        "output_preview": result[:150],
    }

    return {
        "answer": result,
        "reasoning": f"Query escalated: {route_reasoning}",
        "tool_calls": state.get("tool_calls", []) + [tool_call],
        "sources": [],
        "num_sources": 0,
    }


def handle_out_of_scope(state: AgentState) -> dict:
    """
    Node: Return a polite out-of-scope response.

    Args:
        state (AgentState): Current graph state.

    Returns:
        dict: Partial state update with out-of-scope answer.
    """
    logger.info("[Node: handle_out_of_scope]")
    return {
        "answer": (
            "I'm only able to answer questions related to Meridian Capital Group "
            "HR policies and employment. For other topics, please use the "
            "appropriate resources."
        ),
        "reasoning": "Query is outside the scope of HR policy assistant.",
        "sources": [],
        "num_sources": 0,
    }


def generate(state: AgentState, llm) -> dict:
    """
    Node: Generate the final answer from accumulated context.

    Args:
        state (AgentState): Current graph state.
        llm: Initialized LangChain chat model.

    Returns:
        dict: Partial state update with answer, reasoning, and sources.
    """
    from src.generation.generator import generate as gen_answer
    from src.generation.generator import parse_agentic_response, synthesize

    query = state["query"]
    context = state.get("context", "")
    retrieved_results = state.get("retrieved_results", [])
    route = state.get("route", "hr_retrieval")

    logger.info(f"[Node: generate] Query: '{query[:60]}'")

    if not context:
        return {
            "answer": (
                "I was unable to find sufficient information to answer your question. "
                "Please contact HR at hr@meridiancapitalgroup.com."
            ),
            "reasoning": "No context available for generation.",
            "sources": [],
            "num_sources": 0,
        }

    # Use synthesis prompt for multi-step, agentic prompt for others
    if route == "multi_step":
        answer = synthesize(llm, query, context)
        reasoning = "Synthesized from multi-step sub-query results."
    else:
        raw = gen_answer(llm, query, context, agentic=True)
        parsed = parse_agentic_response(raw)
        answer = parsed["answer"]
        reasoning = parsed["reasoning"]

    # Format sources from retrieved results
    sources = [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page_label": doc.metadata.get("page_label", "unknown"),
            "similarity_score": round(score, 4),
            "text_preview": doc.page_content[:200].replace("\n", " ") + "...",
        }
        for doc, score in retrieved_results
    ]

    return {
        "answer": answer,
        "reasoning": reasoning,
        "sources": sources,
        "num_sources": len(sources),
    }


def update_memory(state: AgentState, memory) -> dict:
    """
    Node: Persist the current turn to conversation memory.

    Args:
        state (AgentState): Current graph state.
        memory (ConversationMemory): Memory instance to update.

    Returns:
        dict: Empty dict (memory is updated as a side effect).
    """
    query = state["query"]
    answer = state.get("answer", "")
    tool_calls = state.get("tool_calls", [])

    logger.info("[Node: update_memory] Persisting turn to memory")

    memory.add_turn(role="user", content=query, tool_calls=tool_calls)
    if answer:
        memory.add_turn(role="assistant", content=answer)

    return {}


# -------------------------------------------------------------------
# Conditional edge functions
# These determine which node to transition to based on current state.
# -------------------------------------------------------------------

def route_to_node(state: AgentState) -> str:
    """
    Conditional edge: Route to the correct node based on classified route.

    Returns:
        str: Name of the next node to execute.
    """
    route = state.get("route", "hr_retrieval")
    route_map = {
        "hr_retrieval": "retrieve",
        "calculation": "calculate",
        "date_lookup": "date_lookup",
        "multi_step": "decompose_multi_step",
        "escalation": "handle_escalation",
        "out_of_scope": "handle_out_of_scope",
    }
    next_node = route_map.get(route, "retrieve")
    logger.info(f"[Edge: route_to_node] route='{route}' → node='{next_node}'")
    return next_node


def after_retrieval(state: AgentState) -> str:
    """
    Conditional edge: After retrieval, decide to re-retrieve or generate.

    Returns:
        str: 'retrieve' to re-retrieve, 'generate' to proceed.
    """
    score = state.get("retrieval_score", 0.0)
    attempts = state.get("retrieval_attempts", 0)
    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", DEFAULT_MIN_RETRIEVAL_SCORE))
    max_retries = int(os.getenv("MAX_AGENT_ITERATIONS", DEFAULT_MAX_RETRIES))

    if score < min_score and attempts <= max_retries:
        logger.info(
            f"[Edge: after_retrieval] Score {score:.3f} < {min_score} — retrying"
        )
        return "retrieve"

    logger.info(
        f"[Edge: after_retrieval] Score {score:.3f} — proceeding to generate"
    )
    return "generate"


def after_tool_or_escalation(state: AgentState) -> str:
    """
    Conditional edge: After non-retrieval tool or escalation/out-of-scope nodes.

    If an answer is already set (escalation/out-of-scope), go to memory update.
    Otherwise proceed to generate.

    Returns:
        str: 'update_memory' or 'generate'
    """
    if state.get("answer"):
        return "update_memory"
    return "generate"


# -------------------------------------------------------------------
# Helper functions for tool parameter extraction
# -------------------------------------------------------------------

def _reformulate_query(llm, query: str) -> str:
    """Ask the LLM to reformulate a query that returned poor results."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Reformulate the following search query to be more specific and likely "
         "to match HR policy document language. Respond with ONLY the new query."),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"query": query}).strip()
    except Exception:
        return query


def _extract_expression(state: AgentState, tools: list) -> str:
    """Extract arithmetic expression from query using the LLM."""
    return _extract_expression_from_query(None, state["query"])


def _extract_expression_from_query(llm, query: str) -> str:
    """Extract arithmetic expression from a query string."""
    if llm is None:
        return "0"
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract the arithmetic expression from this question. "
         "Respond with ONLY the expression (e.g. '120000 * 0.15')."),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"query": query}).strip()
    except Exception:
        return "0"


def _extract_date_params(state: AgentState, tools: list) -> tuple[str, int]:
    """Extract date action and days from query state."""
    return _extract_date_params_from_query(None, state["query"])


def _extract_date_params_from_query(llm, query: str) -> tuple[str, int]:
    """Extract date action and days from a query string."""
    if llm is None:
        return "today", 0
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract the date action and number from this question.\n"
         "Respond with ONLY: action|days\n"
         "Valid actions: today, add_days, subtract_days, add_months\n"
         "Example: 'add_days|90'"),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"query": query}).strip()
        parts = raw.split("|")
        action = parts[0].strip() if parts else "today"
        days = int(parts[1].strip()) if len(parts) > 1 else 0
        return action, days
    except Exception:
        return "today", 0


# -------------------------------------------------------------------
# Graph builder
# -------------------------------------------------------------------

def build_graph(llm, vectorstore, memory):
    """
    Build and compile the LangGraph StateGraph for the agentic pipeline.

    Wires all nodes and conditional edges together into a compiled
    graph that can be invoked with an AgentState dict.

    Args:
        llm: Initialized LangChain chat model.
        vectorstore: Initialized LangChain Pinecone vector store.
        memory (ConversationMemory): Conversation memory instance.

    Returns:
        CompiledGraph: Ready-to-invoke LangGraph compiled state graph.
    """
    from src.agents.router import QueryRouter
    from src.agents.tools import get_tools
    import functools

    router = QueryRouter(llm)
    tools = get_tools(vectorstore)

    # Partially apply dependencies to node functions
    _route_query = functools.partial(route_query, router=router)
    _retrieve = functools.partial(retrieve, vectorstore=vectorstore)
    _check_retrieval = functools.partial(check_retrieval, router=router)
    _calculate = functools.partial(calculate, tools=tools)
    _date_lookup = functools.partial(date_lookup, tools=tools)
    _decompose = functools.partial(
        decompose_multi_step, router=router, vectorstore=vectorstore
    )
    _generate = functools.partial(generate, llm=llm)
    _update_memory = functools.partial(update_memory, memory=memory)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("route_query", _route_query)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("check_retrieval", _check_retrieval)
    graph.add_node("calculate", _calculate)
    graph.add_node("date_lookup", _date_lookup)
    graph.add_node("decompose_multi_step", _decompose)
    graph.add_node("handle_escalation", handle_escalation)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)
    graph.add_node("generate", _generate)
    graph.add_node("update_memory", _update_memory)

    # Entry point
    graph.add_edge(START, "route_query")

    # Conditional routing from route_query to tool nodes
    graph.add_conditional_edges(
        "route_query",
        route_to_node,
        {
            "retrieve": "retrieve",
            "calculate": "calculate",
            "date_lookup": "date_lookup",
            "decompose_multi_step": "decompose_multi_step",
            "handle_escalation": "handle_escalation",
            "handle_out_of_scope": "handle_out_of_scope",
        },
    )

    # After retrieval — check quality, potentially re-retrieve
    graph.add_edge("retrieve", "check_retrieval")
    graph.add_conditional_edges(
        "check_retrieval",
        after_retrieval,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # After non-retrieval tools — go to generate or memory
    graph.add_conditional_edges(
        "calculate",
        after_tool_or_escalation,
        {"generate": "generate", "update_memory": "update_memory"},
    )
    graph.add_conditional_edges(
        "date_lookup",
        after_tool_or_escalation,
        {"generate": "generate", "update_memory": "update_memory"},
    )
    graph.add_conditional_edges(
        "decompose_multi_step",
        after_tool_or_escalation,
        {"generate": "generate", "update_memory": "update_memory"},
    )
    graph.add_conditional_edges(
        "handle_escalation",
        after_tool_or_escalation,
        {"generate": "generate", "update_memory": "update_memory"},
    )
    graph.add_conditional_edges(
        "handle_out_of_scope",
        after_tool_or_escalation,
        {"generate": "generate", "update_memory": "update_memory"},
    )

    # After generation — always update memory then end
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)

    logger.info("LangGraph StateGraph compiled successfully")
    return graph.compile()


# -------------------------------------------------------------------
# Quick test
# python -m src.agents.agent
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model
    from src.vectorstore.store import get_vector_store
    from src.generation.generator import get_llm
    from src.memory.memory import ConversationMemory
    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "How many PTO days do I get?"

    try:
        raw_docs = load_documents(test_path)
        lc_docs = chunk_documents(raw_docs)
        embed_model = get_embed_model()
        vectorstore = get_vector_store(documents=lc_docs, embed_model=embed_model)
        llm = get_llm()
        memory = ConversationMemory()

        graph = build_graph(llm, vectorstore, memory)

        initial_state: AgentState = {
            "query": test_query,
            "route": "",
            "route_reasoning": "",
            "route_confidence": 0.0,
            "context": "",
            "retrieved_results": [],
            "retrieval_score": 0.0,
            "retrieval_attempts": 0,
            "tool_output": "",
            "sub_queries": [],
            "sub_query_results": [],
            "raw_answer": "",
            "answer": "",
            "reasoning": "",
            "tool_calls": [],
            "memory_context": memory.get_context_string(),
            "sources": [],
            "num_sources": 0,
            "iterations": 0,
            "error": None,
        }

        result = graph.invoke(initial_state)

        print(f"\n--- LangGraph Agent Test ---")
        print(f"Query      : {result['query']}")
        print(f"Route      : {result['route']}")
        print(f"Iterations : {result['iterations']}")
        print(f"Reasoning  : {result['reasoning'][:300]}")
        print(f"Answer     : {result['answer']}")
        print(f"Sources    : {result['num_sources']}")
        print(f"Tool calls : {len(result['tool_calls'])}")
        for tc in result["tool_calls"]:
            status = "✓" if tc["success"] else "✗"
            print(f"  [{status}] {tc['tool']}")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
