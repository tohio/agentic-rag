"""
generator.py
------------
Responsible for generating answers from retrieved context using
LangChain's chat model interface and prompt templates.

Key differences from rag-pipeline/generator.py:
    1. Uses LangChain's ChatOpenAI / ChatAnthropic instead of
       LlamaIndex's LLM wrappers
    2. Uses LangChain's ChatPromptTemplate and LCEL (|) pipeline
       syntax for prompt construction
    3. generate_with_reasoning() returns structured reasoning steps
       formatted for display in the LangGraph agent and Streamlit UI

Supported LLMs (configurable via LLM_PROVIDER in .env):
    - openai   : OpenAI GPT-4o (default)
    - anthropic: Anthropic Claude

Configuration (in order of precedence):
    1. Environment variables (LLM_PROVIDER, OPENAI_API_KEY,
       ANTHROPIC_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS)
    2. Sensible defaults

Production note:
    In production, use LangChain's streaming interface via
    .stream() or .astream() for token-by-token response delivery.

Usage:
    from src.generation.generator import get_llm, generate
    llm = get_llm()
    answer = generate(llm, query, context)
"""

import logging
import os
from typing import Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024

# -------------------------------------------------------------------
# Prompt templates
# -------------------------------------------------------------------

# Standard RAG prompt — grounded single-pass answer
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful HR assistant for Meridian Capital Group. "
        "Answer the employee's question using ONLY the context provided. "
        "Be concise, accurate, and professional.\n\n"
        "If the context does not contain enough information, say: "
        "'I don't have enough information in the provided documents to answer that. "
        "Please contact HR at hr@meridiancapitalgroup.com for assistance.'\n\n"
        "Do not make up information or draw from knowledge outside the context."
    )),
    ("human", (
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{query}"
    )),
])

# Agentic reasoning prompt — instructs LLM to show its work
AGENTIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent HR assistant for Meridian Capital Group. "
        "Think step by step before answering. Use ONLY the provided context.\n\n"
        "Format your response as:\n"
        "REASONING:\n<step by step reasoning>\n\n"
        "ANSWER:\n<final answer>"
    )),
    ("human", (
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{query}"
    )),
])

# Synthesis prompt — combines results from multiple tool calls
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent HR assistant for Meridian Capital Group. "
        "You have gathered information from multiple sources to answer a question. "
        "Synthesize all the gathered information into a single, clear, accurate answer. "
        "Be concise and professional."
    )),
    ("human", (
        "ORIGINAL QUESTION:\n{query}\n\n"
        "GATHERED INFORMATION:\n{context}\n\n"
        "Provide a synthesized final answer:"
    )),
])


def get_llm(
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    Initialize and return the configured LangChain chat model.

    Args:
        provider (Optional[str]): LLM provider ('openai' or 'anthropic').
        temperature (Optional[float]): Sampling temperature.
        max_tokens (Optional[int]): Max output tokens.

    Returns:
        BaseChatModel: Configured LangChain chat model instance.

    Raises:
        EnvironmentError: If the required API key is not set.
        ValueError: If an unsupported provider is specified.
    """
    resolved_provider = (
        provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    ).lower()

    resolved_temp = temperature or float(
        os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    resolved_max_tokens = max_tokens or int(
        os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )

    logger.info(
        f"Initializing LangChain LLM — provider: {resolved_provider} | "
        f"temperature: {resolved_temp} | max_tokens: {resolved_max_tokens}"
    )

    if resolved_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL_OPENAI)
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
        )
        logger.info(f"ChatOpenAI ready: {model}")
        return llm

    elif resolved_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Please add it to your .env file."
            )
        model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL_ANTHROPIC)
        llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
        )
        logger.info(f"ChatAnthropic ready: {model}")
        return llm

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{resolved_provider}'. "
            f"Supported: openai, anthropic"
        )


def generate(
    llm,
    query: str,
    context: str,
    agentic: bool = False,
) -> str:
    """
    Generate an answer from retrieved context using LangChain LCEL.

    Builds a prompt | llm | output_parser chain using LangChain's
    pipe syntax (LCEL). This is the standard LangChain pattern and
    integrates cleanly with LangGraph's state graph nodes.

    Args:
        llm: Initialized LangChain chat model from get_llm().
        query (str): The user's natural language question.
        context (str): Formatted retrieved context from retriever.py.
        agentic (bool): If True, uses the agentic reasoning prompt.

    Returns:
        str: Generated answer string.

    Raises:
        ValueError: If query or context is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = AGENTIC_PROMPT if agentic else RAG_PROMPT

    # LCEL chain: prompt | llm | output parser
    chain = prompt | llm | StrOutputParser()

    logger.info(
        f"Generating {'agentic' if agentic else 'standard'} "
        f"answer for: '{query[:60]}'"
    )

    answer = chain.invoke({"query": query.strip(), "context": context})
    logger.info(f"Answer generated ({len(answer)} chars)")
    return answer


def synthesize(
    llm,
    query: str,
    gathered_context: str,
) -> str:
    """
    Synthesize a final answer from multiple tool call results.

    Used by the LangGraph agent after multi-step execution to combine
    results from different tools (retrieval, calculation, date lookup)
    into a single coherent answer.

    Args:
        llm: Initialized LangChain chat model.
        query (str): Original user query.
        gathered_context (str): Combined results from all tool calls.

    Returns:
        str: Synthesized final answer.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not gathered_context or not gathered_context.strip():
        raise ValueError("Gathered context cannot be empty.")

    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()

    logger.info(f"Synthesizing final answer for: '{query[:60]}'")
    answer = chain.invoke({
        "query": query.strip(),
        "context": gathered_context,
    })
    logger.info(f"Synthesis complete ({len(answer)} chars)")
    return answer


def parse_agentic_response(raw_response: str) -> dict:
    """
    Parse an agentic prompt response into reasoning and answer parts.

    The agentic prompt instructs the LLM to format its response as:
        REASONING: <reasoning steps>
        ANSWER: <final answer>

    This function splits that response for display in the Streamlit UI.

    Args:
        raw_response (str): Raw LLM response from generate(agentic=True).

    Returns:
        dict: Contains 'reasoning' and 'answer' keys.
    """
    try:
        reasoning = ""
        answer = raw_response

        if "REASONING:" in raw_response and "ANSWER:" in raw_response:
            parts = raw_response.split("ANSWER:")
            answer = parts[-1].strip()
            reasoning_part = parts[0]
            if "REASONING:" in reasoning_part:
                reasoning = reasoning_part.split("REASONING:")[-1].strip()

        return {"reasoning": reasoning, "answer": answer}

    except Exception:
        return {"reasoning": "", "answer": raw_response}


def build_response(
    query: str,
    answer: str,
    retrieved_results: list,
    reasoning: str = "",
    tool_calls: list = None,
) -> dict:
    """
    Build a structured response dict for the pipeline and UI.

    Args:
        query (str): Original user query.
        answer (str): Final generated answer.
        retrieved_results (list): List of (Document, score) tuples.
        reasoning (str): Agent reasoning steps.
        tool_calls (list): Tool call history from the agent loop.

    Returns:
        dict: Structured response with all components for UI display.
    """
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
        "query": query,
        "answer": answer,
        "reasoning": reasoning,
        "tool_calls": tool_calls or [],
        "sources": sources,
        "num_sources": len(sources),
    }


# -------------------------------------------------------------------
# Quick test
# python -m src.generation.generator
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents
    from src.embedding.embedder import get_embed_model
    from src.vectorstore.store import get_vector_store
    from src.retrieval.retriever import retrieve_with_metadata
    load_dotenv()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "How many PTO days do I get?"

    try:
        raw_docs = load_documents(test_path)
        lc_docs = chunk_documents(raw_docs)
        embed_model = get_embed_model()
        vectorstore = get_vector_store(documents=lc_docs, embed_model=embed_model)
        retrieval = retrieve_with_metadata(vectorstore, test_query)

        llm = get_llm()
        raw = generate(llm, test_query, retrieval["context"], agentic=True)
        parsed = parse_agentic_response(raw)
        response = build_response(
            query=test_query,
            answer=parsed["answer"],
            retrieved_results=retrieval["results"],
            reasoning=parsed["reasoning"],
        )

        print(f"\n--- Generator Test Results ---")
        print(f"Query     : {response['query']}")
        print(f"Reasoning : {response['reasoning'][:300]}...")
        print(f"Answer    : {response['answer']}")
        print(f"Sources   : {response['num_sources']}")

    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)
