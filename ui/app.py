"""
app.py
------
Streamlit UI for the Meridian Capital Group Agentic RAG pipeline
powered by LangChain + LangGraph.

Updated from the previous version to surface LangGraph-specific
response fields:
    - route_confidence   : how certain the router was
    - route_reasoning    : why that route was chosen
    - sub_queries        : decomposed sub-queries for multi_step route
    - graph diagram      : visual ASCII representation of the StateGraph

Usage:
    streamlit run ui/app.py
"""

import json
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Meridian Capital Group — Agentic HR Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Route display config
# -------------------------------------------------------------------
ROUTE_COLORS = {
    "hr_retrieval": "🔵",
    "calculation": "🟡",
    "date_lookup": "🟠",
    "multi_step": "🟣",
    "escalation": "🔴",
    "out_of_scope": "⚫",
}

ROUTE_LABELS = {
    "hr_retrieval": "HR Policy Retrieval",
    "calculation": "Calculation",
    "date_lookup": "Date Lookup",
    "multi_step": "Multi-Step Reasoning",
    "escalation": "Escalation",
    "out_of_scope": "Out of Scope",
}

EXAMPLE_QUESTIONS = [
    "How many PTO days do I get in my first year?",
    "What is 15% of a $120,000 salary?",
    "When will my 90-day PTO eligibility end if I started today?",
    "How much will my VP bonus be and when is it paid?",
    "What are the steps in the disciplinary process?",
    "How do RSUs vest for eligible employees?",
    "What is my current salary?",
    "What is the weather today?",
]


# -------------------------------------------------------------------
# Pipeline initialization
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing LangGraph pipeline...")
def load_pipeline():
    try:
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        return pipeline, None
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        return None, str(e)


def init_session_state():
    defaults = {
        "messages": [],
        "last_response": None,
        "total_queries": 0,
        "total_tool_calls": 0,
        "route_counts": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# -------------------------------------------------------------------
# UI Components
# -------------------------------------------------------------------

def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown("## 🏦 Meridian Capital Group")
        st.markdown("**Agentic HR Assistant**")
        st.markdown("*LangChain + LangGraph*")
        st.divider()

        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Queries", st.session_state.total_queries)
        col2.metric("Tool Calls", st.session_state.total_tool_calls)

        if st.session_state.route_counts:
            st.markdown("**Routes used:**")
            for route, count in st.session_state.route_counts.items():
                icon = ROUTE_COLORS.get(route, "⚪")
                label = ROUTE_LABELS.get(route, route)
                st.markdown(f"{icon} {label}: **{count}**")

        st.divider()

        st.markdown("### 🧠 Memory")
        if pipeline:
            state = pipeline.memory.to_dict()
            st.markdown(f"Turns: **{state['turn_count']}**")
            if state["has_summary"]:
                st.markdown("*(older turns summarized)*")
                with st.expander("View summary"):
                    st.text(state["summary_preview"])

            if st.button("🗑️ Reset Memory", use_container_width=True):
                pipeline.reset_memory()
                st.session_state.messages = []
                st.session_state.last_response = None
                st.success("Memory cleared.")
                st.rerun()

        st.divider()

        st.markdown("### 💡 Examples")
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["pending_query"] = ex
                st.rerun()

        st.divider()
        st.markdown(
            "**Stack**\n"
            "- LLM: GPT-4o\n"
            "- Embeddings: text-embedding-3-small\n"
            "- Vector DB: Pinecone\n"
            "- Agent: LangGraph StateGraph\n"
            "- Framework: LangChain\n"
            "- UI: Streamlit"
        )


def render_agent_trace(response: dict):
    """Render the LangGraph agent trace with all state fields."""
    with st.expander("🔍 LangGraph Agent Trace", expanded=True):

        # Route decision row
        route = response.get("route", "unknown")
        confidence = response.get("route_confidence", 0.0)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(
            f"**Route**\n\n"
            f"{ROUTE_COLORS.get(route, '⚪')} {ROUTE_LABELS.get(route, route)}"
        )
        col2.markdown(f"**Confidence**\n\n{confidence:.0%}")
        col3.markdown(f"**Iterations**\n\n{response.get('iterations', 0)}")
        col4.markdown(f"**Sources**\n\n{response.get('num_sources', 0)} chunks")

        # Route reasoning (new — from LangGraph structured output)
        if response.get("route_reasoning"):
            st.caption(f"*Routing reason: {response['route_reasoning']}*")

        st.divider()

        # Sub-queries (new — from multi_step decomposition)
        if response.get("sub_queries"):
            st.markdown("**Sub-queries (multi-step decomposition):**")
            for i, sq in enumerate(response["sub_queries"], 1):
                st.markdown(f"`{i}.` {sq}")
            st.divider()

        # Reasoning steps
        if response.get("reasoning"):
            st.markdown("**Reasoning:**")
            for i, step in enumerate(response["reasoning"].split("\n"), 1):
                if step.strip():
                    st.markdown(f"`{i}.` {step.strip()}")

        # Tool calls
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            st.divider()
            st.markdown("**Tool Calls:**")
            for tc in tool_calls:
                icon = "✅" if tc["success"] else "❌"
                with st.expander(
                    f"{icon} `{tc['tool']}` — "
                    f"{'Success' if tc['success'] else 'Failed'}"
                ):
                    if tc.get("output_preview"):
                        st.text(tc["output_preview"])

        # Sources
        sources = response.get("sources", [])
        if sources:
            st.divider()
            st.markdown("**Retrieved Sources:**")
            for i, src in enumerate(sources, 1):
                st.markdown(
                    f"{i}. `{src['file_name']}` — "
                    f"Page {src['page_label']} "
                    f"*(score: {src['similarity_score']})*"
                )
                with st.expander(f"Preview chunk {i}"):
                    st.text(src.get("text_preview", ""))


def render_chat_tab(pipeline):
    st.markdown("### 💬 Ask the HR Assistant")
    st.markdown(
        "Powered by a LangGraph StateGraph — routing, tool use, "
        "multi-step reasoning, and memory in a compiled state graph."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask a question about HR policies...") or pending

    if user_input and pipeline:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("LangGraph is reasoning..."):
                try:
                    response = pipeline.query(user_input)
                    st.markdown(response["answer"])

                    st.session_state.total_queries += 1
                    st.session_state.total_tool_calls += len(
                        response.get("tool_calls", [])
                    )
                    route = response.get("route", "unknown")
                    st.session_state.route_counts[route] = (
                        st.session_state.route_counts.get(route, 0) + 1
                    )
                    st.session_state.last_response = response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["answer"]}
                    )
                except Exception as e:
                    err = f"Something went wrong: {e}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )

    if st.session_state.last_response:
        st.divider()
        render_agent_trace(st.session_state.last_response)


def render_comparison_tab(pipeline):
    st.markdown("### ⚖️ Standard RAG vs LangGraph Agentic RAG")
    st.markdown(
        "Compare a single retrieve-generate pass with the full "
        "LangGraph state graph — routing, tools, and reasoning."
    )

    compare_input = st.text_area(
        "Question to compare:",
        placeholder="How much will my VP bonus be and when is it paid?",
        height=80,
    )

    if st.button("Compare", type="primary") and compare_input and pipeline:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 Standard RAG")
            st.markdown("*Single retrieve → generate pass*")
            with st.spinner("Running..."):
                try:
                    from src.retrieval.retriever import retrieve_with_metadata
                    from src.generation.generator import get_llm, generate

                    llm = get_llm()
                    retrieval = retrieve_with_metadata(
                        pipeline.vectorstore, compare_input
                    )
                    answer = generate(llm, compare_input, retrieval["context"])
                    st.success(answer)
                    st.caption(
                        f"Retrieved {retrieval['num_results']} chunk(s) | "
                        f"max score: {retrieval['max_score']}"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("#### 🤖 LangGraph Agentic RAG")
            st.markdown("*Full StateGraph: route → tools → generate*")
            with st.spinner("Running LangGraph..."):
                try:
                    response = pipeline.query(compare_input)
                    st.success(response["answer"])
                    route = response["route"]
                    st.caption(
                        f"{ROUTE_COLORS.get(route, '⚪')} "
                        f"Route: {ROUTE_LABELS.get(route, route)} | "
                        f"Confidence: {response.get('route_confidence', 0):.0%} | "
                        f"Iterations: {response['iterations']}"
                    )
                    render_agent_trace(response)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("**Good examples to try:**")
    for ex in [
        "How much will my VP bonus be and when is it paid?",
        "When will I be eligible for PTO if I start today?",
        "What is 30% of a $150,000 Managing Director salary?",
    ]:
        st.code(ex)


def render_graph_tab(pipeline):
    """Render the LangGraph structure visualization tab."""
    st.markdown("### 🗺️ LangGraph State Graph")
    st.markdown(
        "The agent is modeled as a compiled `StateGraph` where nodes are "
        "discrete actions and edges are conditional transitions."
    )

    if pipeline:
        st.code(pipeline.get_graph_diagram(), language="text")

    st.markdown("#### Node Descriptions")
    nodes = {
        "route_query": "Classifies the query using structured LLM output (RouteDecision). Runs escalation pre-filter first.",
        "retrieve": "Calls hr_policy_retriever tool. Fetches top-k chunks from Pinecone by cosine similarity.",
        "check_retrieval": "Inspects retrieval score. Re-routes back to retrieve with reformulated query if score is low.",
        "calculate": "Calls calculator tool. Extracts arithmetic expression from query using LLM, then evaluates safely.",
        "date_lookup": "Calls date_calculator tool. Extracts action and days from query, computes relative date.",
        "decompose_multi_step": "Decomposes complex query into sub-queries, routes and executes each, accumulates context.",
        "handle_escalation": "Calls escalation_router tool. Returns HR contact info when handbook is insufficient.",
        "handle_out_of_scope": "Returns polite out-of-scope message for unrelated queries.",
        "generate": "Generates final answer using LCEL chain (prompt | llm | parser). Uses synthesis prompt for multi-step.",
        "update_memory": "Persists user query and assistant answer to ConversationMemory. Always runs last.",
    }
    for node, desc in nodes.items():
        with st.expander(f"`{node}`"):
            st.markdown(desc)


def render_metrics_tab():
    st.markdown("### 📊 Evaluation Metrics")
    results_path = "evaluation/results.json"

    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_data = json.load(f)
        metrics = eval_data.get("metrics", {})

        st.markdown("#### Answer Quality")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hit Rate", f"{metrics.get('hit_rate', 0) * 100:.1f}%")
        c2.metric("Faithfulness", f"{metrics.get('avg_faithfulness', 0):.2f} / 5")
        c3.metric("Relevance", f"{metrics.get('avg_relevance', 0):.2f} / 5")
        c4.metric("Correctness", f"{metrics.get('avg_correctness', 0):.2f} / 5")

        st.markdown("#### LangGraph Agent Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route Accuracy", f"{metrics.get('route_accuracy', 0) * 100:.1f}%")
        c2.metric("Tool Precision", f"{metrics.get('tool_precision', 0) * 100:.1f}%")
        c3.metric("Avg Iterations", f"{metrics.get('avg_iterations', 0):.1f}")
        c4.metric("Escalation Rate", f"{metrics.get('escalation_rate', 0) * 100:.1f}%")

        st.markdown("#### By Difficulty")
        for difficulty, score in metrics.get("by_difficulty", {}).items():
            st.progress(score / 5, text=f"{difficulty}: {score:.2f} / 5")

        st.markdown("#### By Route")
        for route, data in metrics.get("by_route", {}).items():
            icon = ROUTE_COLORS.get(route, "⚪")
            label = ROUTE_LABELS.get(route, route)
            st.markdown(
                f"{icon} **{label}** — "
                f"{data['count']} questions | "
                f"avg correctness: {data['avg_correctness']:.2f} / 5"
            )
    else:
        st.info("No results yet. Run `python evaluation/eval.py --output evaluation/results.json`")
        st.code("python evaluation/eval.py --output evaluation/results.json")


def render_about_tab(pipeline):
    st.markdown("""
    ## About This Project

    An **Agentic RAG pipeline** built with **LangChain + LangGraph** over a fictional
    HR handbook for Meridian Capital Group.

    ### Architecture

    | Component | Technology |
    |---|---|
    | Agent loop | LangGraph `StateGraph` |
    | LLM | LangChain `ChatOpenAI` (GPT-4o) |
    | Embeddings | LangChain `OpenAIEmbeddings` |
    | Vector store | LangChain `PineconeVectorStore` |
    | Tools | LangChain `@tool` decorator |
    | Routing | LangChain `with_structured_output` |
    | Memory | Custom sliding window + summarization |
    | PDF loading | LlamaIndex `SimpleDirectoryReader` |
    | UI | Streamlit |

    ### vs Standard RAG (rag-pipeline)

    | Feature | rag-pipeline | agentic-rag |
    |---|---|---|
    | Framework | LlamaIndex | LangChain + LangGraph |
    | Agent loop | None | LangGraph `StateGraph` |
    | Routing | None | Structured output router |
    | Tools | None | 4 tools via `@tool` |
    | Memory | None | Sliding window + summarization |
    | Vector store | Chroma (local) | Pinecone (cloud) |
    | UI | Gradio | Streamlit |

    ### Production Considerations
    - **Memory** → Redis for multi-user persistent session storage
    - **API layer** → FastAPI with async LangGraph streaming
    - **Observability** → LangSmith for full graph trace logging
    - **Vector store** → Pinecone namespaces for multi-tenant isolation

    [rag-pipeline repo](https://github.com/yourusername/rag-pipeline) |
    [agentic-rag repo](https://github.com/yourusername/agentic-rag)
    """)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    init_session_state()
    pipeline, init_error = load_pipeline()

    if init_error:
        st.error(f"Pipeline failed to initialize: {init_error}")
        st.code("cp .env.example .env\n# Fill in your API keys")
        return

    render_sidebar(pipeline)

    st.title("🏦 Meridian Capital Group — Agentic HR Assistant")
    st.markdown(
        "LangChain + LangGraph | Query routing · Tool use · "
        "Multi-step reasoning · Conversation memory"
    )
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Ask HR",
        "⚖️ RAG vs Agentic",
        "🗺️ Graph Structure",
        "📊 Evaluation",
        "ℹ️ About",
    ])

    with tab1:
        render_chat_tab(pipeline)
    with tab2:
        render_comparison_tab(pipeline)
    with tab3:
        render_graph_tab(pipeline)
    with tab4:
        render_metrics_tab()
    with tab5:
        render_about_tab(pipeline)


if __name__ == "__main__":
    main()
