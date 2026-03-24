"""
tests/test_tools.py

Unit tests for all CrewAI tools — run without network access or API keys.
Tests cover input schema validation, tool instantiation, helper function
logic, and graceful error handling on missing credentials.

Run:
    pytest tests/test_tools.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import sys
import types
import re

# ── Stub crewai and external deps before importing tools ──────────
crewai_mod = types.ModuleType("crewai")
crewai_tools_mod = types.ModuleType("crewai.tools")

class BaseTool:
    name: str = ""
    description: str = ""
    args_schema = None

crewai_tools_mod.BaseTool = BaseTool
sys.modules.setdefault("crewai", crewai_mod)
sys.modules.setdefault("crewai.tools", crewai_tools_mod)

# Stub tavily
tavily_mod = types.ModuleType("tavily")
class TavilyClient:
    def __init__(self, api_key): pass
    def search(self, **kwargs): return {"results": []}
tavily_mod.TavilyClient = TavilyClient
sys.modules.setdefault("tavily", tavily_mod)

sys.path.insert(0, ".")


# ══════════════════════════════════════════════════════════════════
# SEC TOOLS
# ══════════════════════════════════════════════════════════════════

from tools.sec_tool import (
    SECFilingSearchTool,
    SECFilingParserTool,
    SECFilingSearchInput,
    SECFilingParserInput,
    KEY_SECTIONS,
    _extract_sections,
)


class TestSECFilingSearchTool:
    def test_instantiation(self):
        tool = SECFilingSearchTool()
        assert tool.name == "sec_filing_search"
        assert "EDGAR" in tool.description or "SEC" in tool.description

    def test_input_schema_defaults(self):
        inp = SECFilingSearchInput(ticker="AAPL")
        assert inp.form_type == "10-K"
        assert inp.num_results == 3

    def test_input_schema_custom(self):
        inp = SECFilingSearchInput(ticker="MSFT", form_type="10-Q", num_results=5)
        assert inp.ticker == "MSFT"
        assert inp.form_type == "10-Q"
        assert inp.num_results == 5

    def test_num_results_bounds(self):
        with pytest.raises(Exception):
            SECFilingSearchInput(ticker="AAPL", num_results=0)
        with pytest.raises(Exception):
            SECFilingSearchInput(ticker="AAPL", num_results=11)

    def test_graceful_error_on_network_failure(self):
        tool = SECFilingSearchTool()
        result = tool._run(ticker="AAPL")
        assert isinstance(result, str)
        assert len(result) > 0


class TestSECFilingParserTool:
    def test_instantiation(self):
        tool = SECFilingParserTool()
        assert tool.name == "sec_filing_parser"

    def test_input_schema_defaults(self):
        inp = SECFilingParserInput(filing_url="https://www.sec.gov/test")
        assert inp.sections is None
        assert inp.max_chars_per_section == 4000

    def test_max_chars_bounds(self):
        with pytest.raises(Exception):
            SECFilingParserInput(filing_url="https://x.com", max_chars_per_section=100)
        with pytest.raises(Exception):
            SECFilingParserInput(filing_url="https://x.com", max_chars_per_section=99999)

    def test_graceful_error_on_bad_url(self):
        tool = SECFilingParserTool()
        result = tool._run(filing_url="https://www.sec.gov/nonexistent")
        assert isinstance(result, str)

    def test_key_sections_defined(self):
        assert "Item 1" in KEY_SECTIONS
        assert "Item 1A" in KEY_SECTIONS
        assert "Item 7" in KEY_SECTIONS
        assert len(KEY_SECTIONS) >= 4

    def test_extract_sections_bullish_html(self):
        html = """<html><body>
        <p>Item 1. Business Overview</p>
        <p>Apple designs consumer electronics and software.</p>
        <p>Item 1A. Risk Factors</p>
        <p>Our business faces significant competition.</p>
        </body></html>"""
        result = _extract_sections(html, ["Item 1", "Item 1A"], max_chars=500)
        assert "Item 1" in result
        assert "Item 1A" in result

    def test_extract_sections_missing_returns_string(self):
        html = "<html><body><p>No sections here.</p></body></html>"
        result = _extract_sections(html, ["Item 7"], max_chars=500)
        assert "Item 7" in result
        assert "not found" in result["Item 7"].lower()

    def test_extract_sections_respects_max_chars(self):
        long_text = "word " * 2000
        html = f"<html><body><p>Item 1. Business</p><p>{long_text}</p><p>Item 1A.</p></body></html>"
        result = _extract_sections(html, ["Item 1"], max_chars=200)
        assert len(result["Item 1"]) <= 200


# ══════════════════════════════════════════════════════════════════
# FINANCIAL TOOLS
# ══════════════════════════════════════════════════════════════════

from tools.financial_tool import (
    StockOverviewTool,
    CompetitorBenchmarkTool,
    AnalystRatingsTool,
    StockOverviewInput,
    CompetitorBenchmarkInput,
    AnalystRatingsInput,
    _fmt,
)


class TestFmtHelper:
    def test_billions(self):
        assert _fmt(1_500_000_000) == "1.50B"

    def test_trillions(self):
        assert _fmt(2_300_000_000_000) == "2.30T"

    def test_millions(self):
        assert _fmt(450_000_000) == "450.00M"

    def test_none_returns_na(self):
        assert _fmt(None) == "N/A"

    def test_nan_string_returns_na(self):
        assert _fmt("nan") == "N/A"

    def test_prefix_and_suffix(self):
        result = _fmt(1_000_000_000, prefix="$", suffix="")
        assert result.startswith("$")

    def test_custom_decimals(self):
        result = _fmt(1_000_000_000, decimals=0)
        assert "." not in result


class TestStockOverviewTool:
    def test_instantiation(self):
        tool = StockOverviewTool()
        assert tool.name == "stock_overview"

    def test_input_schema_defaults(self):
        inp = StockOverviewInput(ticker="AAPL")
        assert inp.period == "1y"

    def test_input_schema_custom_period(self):
        inp = StockOverviewInput(ticker="NVDA", period="6mo")
        assert inp.period == "6mo"

    def test_graceful_error_returns_string(self):
        tool = StockOverviewTool()
        result = tool._run(ticker="AAPL")
        assert isinstance(result, str)


class TestCompetitorBenchmarkTool:
    def test_instantiation(self):
        tool = CompetitorBenchmarkTool()
        assert tool.name == "competitor_benchmark"

    def test_metric_map_completeness(self):
        tool = CompetitorBenchmarkTool()
        required = [
            "pe_ratio", "pb_ratio", "ev_ebitda", "profit_margin",
            "revenue_growth", "market_cap", "debt_to_equity", "return_on_equity",
        ]
        for m in required:
            assert m in tool.METRIC_MAP, f"Missing metric: {m}"

    def test_input_schema_validation(self):
        inp = CompetitorBenchmarkInput(ticker="AAPL", competitors=["MSFT", "GOOGL"])
        assert len(inp.competitors) == 2

    def test_competitors_max_length(self):
        with pytest.raises(Exception):
            CompetitorBenchmarkInput(
                ticker="AAPL",
                competitors=["A", "B", "C", "D", "E", "F", "G", "H", "I"],  # 9 — exceeds max 8
            )

    def test_graceful_error_returns_string(self):
        tool = CompetitorBenchmarkTool()
        result = tool._run(ticker="AAPL", competitors=["MSFT"])
        assert isinstance(result, str)


class TestAnalystRatingsTool:
    def test_instantiation(self):
        tool = AnalystRatingsTool()
        assert tool.name == "analyst_ratings"

    def test_input_schema_defaults(self):
        inp = AnalystRatingsInput(ticker="NVDA")
        assert inp.include_eps_estimates == True

    def test_graceful_error_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        tool = AnalystRatingsTool()
        result = tool._run(ticker="AAPL")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# SEARCH / SENTIMENT TOOLS
# ══════════════════════════════════════════════════════════════════

from tools.search_tool import (
    NewsSearchTool,
    SentimentTool,
    NewsSearchInput,
    SentimentInput,
    _score_sentiment,
    _build_queries,
    BULLISH_WORDS,
    BEARISH_WORDS,
)


class TestScoreSentiment:
    def test_bullish_text(self):
        b, bear, label = _score_sentiment("Apple beats earnings, stock surges to record high")
        assert b > bear
        assert "BULLISH" in label

    def test_bearish_text(self):
        b, bear, label = _score_sentiment("Company misses revenue, faces layoffs and investigation")
        assert bear > b
        assert "BEARISH" in label

    def test_neutral_text(self):
        _, _, label = _score_sentiment("Company reports quarterly update")
        assert label == "NEUTRAL ➡️"

    def test_empty_string(self):
        b, bear, label = _score_sentiment("")
        assert b == 0
        assert bear == 0


class TestBuildQueries:
    def test_default_topics(self):
        queries = _build_queries("Apple Inc.", "AAPL", None)
        assert len(queries) == 5
        assert all("Apple Inc." in q for q in queries)

    def test_custom_topics(self):
        queries = _build_queries("Apple Inc.", "AAPL", ["AI strategy", "China sales"])
        assert len(queries) == 2
        assert "AI strategy" in queries[0]
        assert "China sales" in queries[1]

    def test_ticker_included(self):
        queries = _build_queries("Tesla Inc.", "TSLA", None)
        assert all("TSLA" in q for q in queries)


class TestKeywordSets:
    def test_bullish_words_populated(self):
        assert len(BULLISH_WORDS) >= 15
        assert "beat" in BULLISH_WORDS
        assert "surge" in BULLISH_WORDS

    def test_bearish_words_populated(self):
        assert len(BEARISH_WORDS) >= 15
        assert "miss" in BEARISH_WORDS
        assert "layoffs" in BEARISH_WORDS

    def test_no_overlap(self):
        overlap = BULLISH_WORDS & BEARISH_WORDS
        assert len(overlap) == 0, f"Overlapping words: {overlap}"


class TestNewsSearchTool:
    def test_instantiation(self):
        tool = NewsSearchTool()
        assert tool.name == "news_search"

    def test_input_schema_defaults(self):
        inp = NewsSearchInput(company_name="Apple Inc.", ticker="AAPL")
        assert inp.days_back == 30
        assert inp.max_results == 10

    def test_days_back_bounds(self):
        with pytest.raises(Exception):
            NewsSearchInput(company_name="X", ticker="X", days_back=0)
        with pytest.raises(Exception):
            NewsSearchInput(company_name="X", ticker="X", days_back=91)

    def test_graceful_error_no_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = NewsSearchTool()
        result = tool._run(company_name="Apple Inc.", ticker="AAPL")
        assert isinstance(result, str)
        assert "Error" in result


class TestSentimentTool:
    def test_instantiation(self):
        tool = SentimentTool()
        assert tool.name == "news_sentiment"

    def test_input_schema_defaults(self):
        inp = SentimentInput(company_name="Apple Inc.", ticker="AAPL")
        assert inp.days_back == 30
        assert inp.num_articles == 15

    def test_num_articles_bounds(self):
        with pytest.raises(Exception):
            SentimentInput(company_name="X", ticker="X", num_articles=4)
        with pytest.raises(Exception):
            SentimentInput(company_name="X", ticker="X", num_articles=31)

    def test_graceful_error_no_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool = SentimentTool()
        result = tool._run(company_name="Apple Inc.", ticker="AAPL")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# MEMO TEMPLATE
# ══════════════════════════════════════════════════════════════════

from output.memo_template import (
    process_memo,
    validate_sections,
    extract_verdict,
    get_section,
    strip_markdown,
    REQUIRED_SECTIONS,
    ProcessedMemo,
)

COMPLETE_MEMO = """
## Executive Summary
Apple Inc. is a global technology leader with strong brand equity.

## Company Overview
Apple designs consumer electronics, software, and services.

## Financial Analysis
Revenue of $391B TTM with 26% net margins.

## Competitive Landscape
Trades at 28x P/E vs peer median of 22x.

## News & Sentiment
Overall sentiment: BULLISH.

## Macro Context
Rising rates are a mild headwind.

## Analyst Consensus
32 analysts: 26 Buy, 5 Hold, 1 Sell.

## Risk Factors
1. China revenue concentration (High)

## Investment Thesis
Bull: Services mix shift. Bear: Hardware saturation.

## Verdict & Recommendation
**BUY** — Strong ecosystem moat supports a $210 price target.
"""


class TestValidateSections:
    def test_complete_memo(self):
        found, missing = validate_sections(COMPLETE_MEMO)
        assert len(found) == 10
        assert len(missing) == 0

    def test_partial_memo(self):
        partial = "## Executive Summary\nContent.\n## Company Overview\nMore."
        found, missing = validate_sections(partial)
        assert len(missing) > 0

    def test_empty_memo(self):
        found, missing = validate_sections("")
        assert len(found) == 0
        assert len(missing) == len(REQUIRED_SECTIONS)


class TestExtractVerdict:
    def test_buy_verdict(self):
        verdict, emoji = extract_verdict(COMPLETE_MEMO)
        assert verdict == "Buy"
        assert emoji == "🟢"

    def test_sell_verdict(self):
        memo = "## Verdict & Recommendation\n**SELL** — Deteriorating fundamentals."
        verdict, emoji = extract_verdict(memo)
        assert verdict is not None
        assert emoji == "🔴"

    def test_hold_verdict(self):
        memo = "## Verdict & Recommendation\n**HOLD** — Fairly valued at current levels."
        verdict, emoji = extract_verdict(memo)
        assert emoji == "🟡"

    def test_no_verdict(self):
        verdict, emoji = extract_verdict("No recommendation here.")
        assert verdict is None
        assert emoji == "⬜"


class TestProcessMemo:
    def test_complete_memo(self):
        result = process_memo(COMPLETE_MEMO, "aapl", "Apple Inc.", runtime_seconds=142.5)
        assert isinstance(result, ProcessedMemo)
        assert result.metadata.ticker == "AAPL"
        assert result.metadata.is_complete == True
        assert result.metadata.verdict == "Buy"
        assert "142s" in result.rendered

    def test_ticker_uppercased(self):
        result = process_memo(COMPLETE_MEMO, "aapl", "Apple Inc.")
        assert result.metadata.ticker == "AAPL"

    def test_incomplete_memo_flagged(self):
        result = process_memo("## Executive Summary\nOnly one section.", "MSFT", "Microsoft")
        assert result.metadata.is_complete == False
        assert len(result.metadata.sections_missing) > 0

    def test_rendered_contains_header(self):
        result = process_memo(COMPLETE_MEMO, "AAPL", "Apple Inc.", runtime_seconds=90.0)
        assert "AAPL" in result.rendered
        assert "Apple Inc." in result.rendered

    def test_plain_text_strips_markdown(self):
        result = process_memo(COMPLETE_MEMO, "AAPL", "Apple Inc.")
        assert "##" not in result.plain_text
        assert "**" not in result.plain_text


class TestStripMarkdown:
    def test_removes_headers(self):
        assert "##" not in strip_markdown("## Hello World")

    def test_removes_bold(self):
        assert "**" not in strip_markdown("**bold text**")

    def test_removes_links(self):
        result = strip_markdown("[click here](https://example.com)")
        assert "https" not in result
        assert "click here" in result

    def test_preserves_content(self):
        result = strip_markdown("## Hello **World**")
        assert "Hello" in result
        assert "World" in result


class TestGetSection:
    def test_existing_section(self):
        section = get_section(COMPLETE_MEMO, "Risk Factors")
        assert section is not None
        assert "China" in section

    def test_missing_section(self):
        section = get_section(COMPLETE_MEMO, "Nonexistent Section")
        assert section is None

    def test_returns_header_and_content(self):
        section = get_section(COMPLETE_MEMO, "Executive Summary")
        assert "Executive Summary" in section
        assert "Apple" in section
