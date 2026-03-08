"""
test_pipeline.py
----------------
Unit tests for the agentic-rag pipeline (LangChain + LangGraph).

Test coverage:
    - Inherited components (loader, chunker, embedder, memory, tools)
    - Updated components (store, retriever, generator with LangChain)
    - New LangGraph components (router structured output, agent state graph)
    - Integration tests (full graph invocation, multi-turn memory)

Usage:
    # All unit tests
    pytest tests/test_pipeline.py -v

    # Exclude slow integration tests
    pytest tests/test_pipeline.py -v -m "not slow"

    # Only LangGraph-specific tests
    pytest tests/test_pipeline.py::TestAgent -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_llama_doc():
    """Minimal LlamaIndex Document for testing the loader/chunker bridge."""
    from llama_index.core.schema import Document
    return Document(
        text=(
            "Meridian Capital Group provides all full-time employees with 15 days "
            "of PTO in their first year of service. PTO accrues monthly."
        ),
        metadata={
            "file_name": "meridian-capital-handbook.pdf",
            "file_path": "/data/raw/meridian-capital-handbook.pdf",
            "page_label": "5",
        },
    )


@pytest.fixture
def sample_llama_docs(sample_llama_doc):
    return [sample_llama_doc]


@pytest.fixture
def sample_lc_docs(sample_llama_docs):
    """LangChain Documents produced by chunker."""
    from src.ingestion.chunker import chunk_documents
    return chunk_documents(sample_llama_docs)


@pytest.fixture
def mock_llm():
    """Mock LangChain chat model."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Employees receive 15 days of PTO."
    mock.invoke.return_value = mock_response
    # with_structured_output returns a chain-like object
    mock.with_structured_output.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_vectorstore():
    """Mock LangChain PineconeVectorStore."""
    from langchain.schema import Document
    mock = MagicMock()
    mock_doc = Document(
        page_content="Employees receive 15 days of PTO.",
        metadata={"file_name": "handbook.pdf", "page_label": "5"},
    )
    mock.similarity_search_with_score.return_value = [(mock_doc, 0.87)]
    mock.as_retriever.return_value = MagicMock()
    return mock


@pytest.fixture
def memory():
    from src.memory.memory import ConversationMemory
    return ConversationMemory(max_turns=6, summary_threshold=4)


# -------------------------------------------------------------------
# loader.py tests (inherited — unchanged from rag-pipeline)
# -------------------------------------------------------------------

class TestLoader:

    def test_load_nonexistent_path_raises(self):
        from src.ingestion.loader import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path/file.pdf")

    def test_load_unsupported_file_type_raises(self, tmp_path):
        from src.ingestion.loader import load_documents
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_documents(str(txt_file))

    def test_load_empty_directory_raises(self, tmp_path):
        from src.ingestion.loader import load_documents
        with pytest.raises(ValueError, match="No PDF files found"):
            load_documents(str(tmp_path))


# -------------------------------------------------------------------
# chunker.py tests — LangChain Document output
# -------------------------------------------------------------------

class TestChunker:

    def test_chunk_returns_langchain_documents(self, sample_llama_docs):
        """Should return LangChain Documents, not LlamaIndex nodes."""
        from src.ingestion.chunker import chunk_documents
        from langchain.schema import Document
        docs = chunk_documents(sample_llama_docs)
        assert len(docs) >= 1
        assert isinstance(docs[0], Document)

    def test_chunk_has_page_content_field(self, sample_llama_docs):
        """LangChain Documents use page_content, not text."""
        from src.ingestion.chunker import chunk_documents
        docs = chunk_documents(sample_llama_docs)
        assert hasattr(docs[0], "page_content")
        assert len(docs[0].page_content) > 0

    def test_chunk_preserves_metadata(self, sample_llama_docs):
        """Metadata should be carried through from LlamaIndex to LangChain."""
        from src.ingestion.chunker import chunk_documents
        docs = chunk_documents(sample_llama_docs)
        assert "file_name" in docs[0].metadata
        assert "page_label" in docs[0].metadata
        assert "source" in docs[0].metadata

    def test_chunk_empty_raises(self):
        from src.ingestion.chunker import chunk_documents
        with pytest.raises(ValueError, match="No documents provided"):
            chunk_documents([])

    def test_chunk_overlap_exceeds_size_raises(self, sample_llama_docs):
        from src.ingestion.chunker import chunk_documents
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            chunk_documents(sample_llama_docs, chunk_size=100, chunk_overlap=100)

    def test_get_chunk_metadata_structure(self, sample_lc_docs):
        from src.ingestion.chunker import get_chunk_metadata
        metadata = get_chunk_metadata(sample_lc_docs)
        assert len(metadata) > 0
        assert "chunk_index" in metadata[0]
        assert "file_name" in metadata[0]
        assert "text_preview" in metadata[0]


# -------------------------------------------------------------------
# embedder.py tests — LangChain OpenAIEmbeddings
# -------------------------------------------------------------------

class TestEmbedder:

    def test_get_embed_model_raises_without_api_key(self, monkeypatch):
        from src.embedding.embedder import get_embed_model
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_embed_model()

    def test_embed_empty_texts_raises(self):
        from src.embedding.embedder import embed_texts
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="No texts provided"):
            embed_texts([], mock_model)

    def test_embed_query_empty_raises(self):
        from src.embedding.embedder import embed_query
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            embed_query("", mock_model)

    def test_embed_texts_calls_embed_documents(self):
        """Should call embed_documents on the LangChain model."""
        from src.embedding.embedder import embed_texts
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 1536]
        result = embed_texts(["test text"], mock_model)
        mock_model.embed_documents.assert_called_once()
        assert len(result) == 1

    def test_embed_query_calls_embed_query(self):
        """Should call embed_query (not embed_documents) for queries."""
        from src.embedding.embedder import embed_query
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1] * 1536
        result = embed_query("What is PTO?", mock_model)
        mock_model.embed_query.assert_called_once_with("What is PTO?")
        assert len(result) == 1536


# -------------------------------------------------------------------
# retriever.py tests — LangChain retriever
# -------------------------------------------------------------------

class TestRetriever:

    def test_retrieve_empty_query_raises(self, mock_vectorstore):
        from src.retrieval.retriever import retrieve
        retriever = mock_vectorstore.as_retriever()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve(retriever, "")

    def test_retrieve_with_scores_empty_query_raises(self, mock_vectorstore):
        from src.retrieval.retriever import retrieve_with_scores
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve_with_scores(mock_vectorstore, "")

    def test_retrieve_with_metadata_returns_dict(self, mock_vectorstore):
        from src.retrieval.retriever import retrieve_with_metadata
        result = retrieve_with_metadata(mock_vectorstore, "What is PTO?")
        assert "query" in result
        assert "results" in result
        assert "num_results" in result
        assert "max_score" in result
        assert "success" in result
        assert "context" in result

    def test_retrieve_with_metadata_success_flag(self, mock_vectorstore):
        """Should set success=True when results are above cutoff."""
        from src.retrieval.retriever import retrieve_with_metadata
        result = retrieve_with_metadata(
            mock_vectorstore, "What is PTO?", similarity_cutoff=0.5
        )
        # Mock returns score 0.87 which is above 0.5
        assert result["success"] is True
        assert result["num_results"] == 1

    def test_format_retrieved_context_empty(self):
        from src.retrieval.retriever import format_retrieved_context
        result = format_retrieved_context([])
        assert "No relevant context" in result

    def test_format_retrieved_context_includes_chunk_label(self, mock_vectorstore):
        from src.retrieval.retriever import format_retrieved_context
        from langchain.schema import Document
        doc = Document(
            page_content="PTO policy text.",
            metadata={"file_name": "handbook.pdf", "page_label": "5"},
        )
        result = format_retrieved_context([(doc, 0.87)])
        assert "Chunk 1" in result
        assert "handbook.pdf" in result
        assert "PTO policy text." in result


# -------------------------------------------------------------------
# generator.py tests — LangChain LCEL chains
# -------------------------------------------------------------------

class TestGenerator:

    def test_get_llm_raises_without_openai_key(self, monkeypatch):
        from src.generation.generator import get_llm
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_llm()

    def test_get_llm_raises_for_unsupported_provider(self, monkeypatch):
        from src.generation.generator import get_llm
        monkeypatch.setenv("LLM_PROVIDER", "cohere")
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm()

    def test_generate_empty_query_raises(self, mock_llm):
        from src.generation.generator import generate
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generate(mock_llm, "", "some context")

    def test_generate_empty_context_raises(self, mock_llm):
        from src.generation.generator import generate
        with pytest.raises(ValueError, match="Context cannot be empty"):
            generate(mock_llm, "What is PTO?", "")

    def test_parse_agentic_response_splits_correctly(self):
        from src.generation.generator import parse_agentic_response
        raw = (
            "REASONING:\nI looked up the PTO policy.\n\n"
            "ANSWER:\nYou receive 15 days of PTO."
        )
        result = parse_agentic_response(raw)
        assert "reasoning" in result
        assert "answer" in result
        assert "15 days" in result["answer"]
        assert "PTO policy" in result["reasoning"]

    def test_parse_agentic_response_fallback(self):
        """Should return full response as answer if format is unexpected."""
        from src.generation.generator import parse_agentic_response
        raw = "You receive 15 days of PTO."
        result = parse_agentic_response(raw)
        assert result["answer"] == raw
        assert result["reasoning"] == ""


# -------------------------------------------------------------------
# memory.py tests (inherited — unchanged)
# -------------------------------------------------------------------

class TestMemory:

    def test_add_turn_user(self, memory):
        memory.add_turn(role="user", content="What is PTO?")
        assert memory.get_turn_count() == 1

    def test_add_turn_invalid_role_raises(self, memory):
        with pytest.raises(ValueError, match="Invalid role"):
            memory.add_turn(role="system", content="You are helpful.")

    def test_get_context_string_empty(self, memory):
        assert memory.get_context_string() == ""

    def test_get_context_string_includes_turns(self, memory):
        memory.add_turn(role="user", content="What is PTO?")
        memory.add_turn(role="assistant", content="15 days.")
        ctx = memory.get_context_string()
        assert "What is PTO?" in ctx
        assert "15 days." in ctx

    def test_clear_resets_state(self, memory):
        memory.add_turn(role="user", content="Test")
        memory.clear()
        assert memory.get_turn_count() == 0
        assert memory.get_context_string() == ""

    def test_summarization_triggered(self, memory):
        for i in range(5):
            memory.add_turn(role="user", content=f"Q{i}")
            memory.add_turn(role="assistant", content=f"A{i}")
        assert memory.get_turn_count() <= memory.max_turns


# -------------------------------------------------------------------
# tools.py tests — LangChain @tool decorated tools
# -------------------------------------------------------------------

class TestTools:

    def test_calculator_basic(self):
        from src.agents.tools import calculator
        result = calculator.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_calculator_salary_percentage(self):
        from src.agents.tools import calculator
        result = calculator.invoke({"expression": "120000 * 0.15"})
        assert "18,000" in result or "18000" in result

    def test_calculator_division_by_zero(self):
        from src.agents.tools import calculator
        result = calculator.invoke({"expression": "10 / 0"})
        assert "zero" in result.lower()

    def test_calculator_unsafe_expression(self):
        from src.agents.tools import calculator
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert "Unsafe" in result

    def test_date_calculator_today(self):
        from src.agents.tools import date_calculator
        result = date_calculator.invoke({"action": "today", "days": 0})
        assert "Today" in result

    def test_date_calculator_add_days(self):
        from src.agents.tools import date_calculator
        result = date_calculator.invoke({"action": "add_days", "days": 90})
        assert "90" in result
        assert "Result" in result

    def test_date_calculator_unknown_action(self):
        from src.agents.tools import date_calculator
        result = date_calculator.invoke({"action": "invalid", "days": 0})
        assert "Unknown action" in result

    def test_escalation_router_returns_contact_info(self):
        from src.agents.tools import escalation_router
        result = escalation_router.invoke({
            "question": "What is my salary?",
            "reason": "Personal data not in handbook.",
        })
        assert "hr@meridiancapitalgroup.com" in result

    def test_get_tools_returns_list(self, mock_vectorstore):
        from src.agents.tools import get_tools
        tools = get_tools(mock_vectorstore)
        assert isinstance(tools, list)
        assert len(tools) == 4

    def test_get_tools_without_vectorstore_excludes_retriever(self):
        from src.agents.tools import get_tools
        tools = get_tools()
        names = [t.name for t in tools]
        assert "hr_policy_retriever" not in names
        assert "calculator" in names

    def test_tools_have_name_and_description(self, mock_vectorstore):
        """All @tool decorated functions should have name and description."""
        from src.agents.tools import get_tools
        tools = get_tools(mock_vectorstore)
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert len(tool.description) > 0


# -------------------------------------------------------------------
# router.py tests — LangChain structured output router
# -------------------------------------------------------------------

class TestRouter:

    def test_classify_returns_route_decision(self, mock_llm):
        """Should return a RouteDecision with required fields."""
        from src.agents.router import QueryRouter, RouteDecision
        from unittest.mock import MagicMock

        mock_decision = RouteDecision(
            route="hr_retrieval",
            reasoning="Question about HR policy.",
            confidence=0.95,
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_decision

        router = QueryRouter(mock_llm)
        # Patch the classifier directly
        router.classifier = MagicMock()
        router.classifier.invoke.return_value = mock_decision

        result = router.classify("What is the PTO policy?")
        assert result.route == "hr_retrieval"
        assert result.confidence == 0.95
        assert result.reasoning != ""

    def test_classify_empty_query_raises(self, mock_llm):
        from src.agents.router import QueryRouter
        router = QueryRouter(mock_llm)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            router.classify("")

    def test_classify_fallback_on_exception(self, mock_llm):
        """Should return fallback route if classifier raises."""
        from src.agents.router import QueryRouter
        from unittest.mock import MagicMock

        router = QueryRouter(mock_llm, fallback_route="hr_retrieval")
        router.classifier = MagicMock()
        router.classifier.invoke.side_effect = Exception("LLM error")

        result = router.classify("Some query")
        assert result.route == "hr_retrieval"
        assert result.confidence == 0.0

    def test_should_escalate_personal_data(self, mock_llm):
        from src.agents.router import QueryRouter
        router = QueryRouter(mock_llm)
        assert router.should_escalate("What is my salary?") is True
        assert router.should_escalate("I want to report my manager") is True

    def test_should_not_escalate_policy_query(self, mock_llm):
        from src.agents.router import QueryRouter
        router = QueryRouter(mock_llm)
        assert router.should_escalate("What is the PTO policy?") is False

    def test_decompose_returns_list(self, mock_llm):
        from src.agents.router import QueryRouter, SubQueryList
        from unittest.mock import MagicMock

        mock_result = SubQueryList(
            sub_queries=[
                "What is the VP bonus percentage?",
                "When is the bonus paid out?",
            ]
        )
        router = QueryRouter(mock_llm)
        router.decomposer = MagicMock()
        router.decomposer.invoke.return_value = mock_result

        result = router.decompose("How much VP bonus and when paid?", "tools")
        assert isinstance(result, list)
        assert len(result) == 2


# -------------------------------------------------------------------
# agent.py tests — LangGraph StateGraph
# -------------------------------------------------------------------

class TestAgent:

    def test_agent_state_has_required_keys(self):
        """AgentState TypedDict should define all required keys."""
        from src.agents.agent import AgentState
        required = {
            "query", "route", "context", "answer",
            "reasoning", "tool_calls", "sources",
            "num_sources", "iterations",
        }
        # TypedDict keys are in __annotations__
        state_keys = set(AgentState.__annotations__.keys())
        assert required.issubset(state_keys)

    def test_route_to_node_mapping(self):
        """route_to_node should map all valid routes to node names."""
        from src.agents.agent import route_to_node
        routes = [
            "hr_retrieval", "calculation", "date_lookup",
            "multi_step", "escalation", "out_of_scope",
        ]
        for route in routes:
            state = {"route": route}
            result = route_to_node(state)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_after_retrieval_retries_on_low_score(self, monkeypatch):
        """Should return 'retrieve' when score is below threshold."""
        from src.agents.agent import after_retrieval
        monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")
        monkeypatch.setenv("MAX_AGENT_ITERATIONS", "2")
        state = {"retrieval_score": 0.2, "retrieval_attempts": 1}
        assert after_retrieval(state) == "retrieve"

    def test_after_retrieval_proceeds_on_good_score(self, monkeypatch):
        """Should return 'generate' when score meets threshold."""
        from src.agents.agent import after_retrieval
        monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.3")
        state = {"retrieval_score": 0.85, "retrieval_attempts": 1}
        assert after_retrieval(state) == "generate"

    def test_after_retrieval_proceeds_after_max_retries(self, monkeypatch):
        """Should proceed to generate after exhausting retry attempts."""
        from src.agents.agent import after_retrieval
        monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.9")
        monkeypatch.setenv("MAX_AGENT_ITERATIONS", "2")
        state = {"retrieval_score": 0.1, "retrieval_attempts": 3}
        assert after_retrieval(state) == "generate"

    def test_handle_escalation_returns_contact_info(self):
        """Escalation node should include HR contact details."""
        from src.agents.agent import handle_escalation
        state = {
            "query": "What is my salary?",
            "route_reasoning": "Personal data query.",
            "tool_calls": [],
        }
        result = handle_escalation(state)
        assert "answer" in result
        assert "hr@meridiancapitalgroup.com" in result["answer"]
        assert result["num_sources"] == 0

    def test_handle_out_of_scope_returns_polite_message(self):
        from src.agents.agent import handle_out_of_scope
        state = {"query": "What is the weather?"}
        result = handle_out_of_scope(state)
        assert "answer" in result
        assert "HR policies" in result["answer"]

    def test_after_tool_or_escalation_skips_generate_if_answer_set(self):
        """Should go to update_memory if answer is already set."""
        from src.agents.agent import after_tool_or_escalation
        state = {"answer": "Already answered."}
        assert after_tool_or_escalation(state) == "update_memory"

    def test_after_tool_or_escalation_goes_to_generate_if_no_answer(self):
        from src.agents.agent import after_tool_or_escalation
        state = {"answer": ""}
        assert after_tool_or_escalation(state) == "generate"


# -------------------------------------------------------------------
# Integration tests (slow — requires API keys + Pinecone + documents)
# -------------------------------------------------------------------

class TestAgenticPipelineIntegration:

    @pytest.mark.slow
    def test_full_pipeline_hr_retrieval(self):
        """End-to-end test for an HR policy question."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        response = pipeline.query("How many PTO days does a new employee receive?")
        assert response["answer"] != ""
        assert "15" in response["answer"]
        assert response["route"] in {"hr_retrieval", "multi_step"}

    @pytest.mark.slow
    def test_full_pipeline_calculation(self):
        """End-to-end test for a calculation query."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        response = pipeline.query("What is 15% of $120,000?")
        assert response["route"] in {"calculation", "multi_step"}
        assert "18,000" in response["answer"] or "18000" in response["answer"]

    @pytest.mark.slow
    def test_full_pipeline_escalation(self):
        """Personal data query should escalate immediately."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        response = pipeline.query("What is my current salary?")
        assert response["route"] == "escalation"
        assert "hr@meridiancapitalgroup.com" in response["answer"]

    @pytest.mark.slow
    def test_full_pipeline_out_of_scope(self):
        """Unrelated query should be handled gracefully."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        response = pipeline.query("What is the capital of France?")
        assert response["route"] == "out_of_scope"

    @pytest.mark.slow
    def test_multi_turn_memory_persists(self):
        """Agent should remember prior context across turns."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        pipeline.query("What is the PTO policy for new employees?")
        response = pipeline.query("And what about sick leave?")
        assert response["answer"] != ""
        assert pipeline.memory.get_turn_count() >= 2

    @pytest.mark.slow
    def test_reset_memory_clears_turns(self):
        """reset_memory should clear all conversation history."""
        from src.pipeline import AgenticRAGPipeline
        pipeline = AgenticRAGPipeline()
        pipeline.query("What is PTO?")
        pipeline.reset_memory()
        assert pipeline.memory.get_turn_count() == 0
