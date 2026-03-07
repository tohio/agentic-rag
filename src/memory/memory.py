"""
memory.py
---------
Responsible for managing conversation memory in the agentic RAG pipeline.

The memory module maintains a running history of the conversation
between the user and the agent. This history is injected into the
LLM context on each turn, enabling multi-turn reasoning without
losing prior context.

Memory strategy: Sliding window with summarization fallback
    - Maintains the full conversation history up to MAX_HISTORY_TURNS
    - When the history exceeds the limit, older turns are summarized
      into a single condensed entry to preserve context while
      preventing the context window from overflowing
    - This is more robust than a fixed truncation strategy, which
      can silently drop important context

Configuration (in order of precedence):
    1. Environment variables (MAX_HISTORY_TURNS, MEMORY_SUMMARY_THRESHOLD)
    2. Sensible defaults (max_turns: 10, summary_threshold: 8)

Production note:
    This implementation uses in-memory storage — conversation history
    lives in a Python object and is lost when the process restarts.
    In a production system, this would be backed by Redis for persistent,
    low-latency session storage across multiple users and requests.
    This is called out in the README as a production consideration.

Usage:
    from src.memory.memory import ConversationMemory

    memory = ConversationMemory()
    memory.add_turn(role="user", content="What is the PTO policy?")
    memory.add_turn(role="assistant", content="You receive 15 days of PTO.")
    context = memory.get_context_string()
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Defaults
# MAX_HISTORY_TURNS     : max number of turns to keep in full detail
# MEMORY_SUMMARY_THRESHOLD: when to trigger summarization of older turns
# -------------------------------------------------------------------
DEFAULT_MAX_HISTORY_TURNS = 10
DEFAULT_SUMMARY_THRESHOLD = 8


@dataclass
class Turn:
    """
    A single conversational turn (one message from user or assistant).

    Attributes:
        role      : 'user' or 'assistant'
        content   : the message text
        timestamp : when the turn was added
        tool_calls: any tool calls made during this turn (agent only)
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_calls: list = field(default_factory=list)


class ConversationMemory:
    """
    Manages conversation history for the agentic RAG pipeline.

    Maintains a sliding window of recent turns with summarization
    of older context. Provides formatted context strings for injection
    into LLM prompts.

    Attributes:
        max_turns (int): Maximum number of turns before summarization.
        summary_threshold (int): Trigger summarization at this turn count.
        _turns (list): Full list of conversation turns.
        _summary (str): Condensed summary of older turns (if any).
    """

    def __init__(
        self,
        max_turns: Optional[int] = None,
        summary_threshold: Optional[int] = None,
    ):
        self.max_turns = max_turns or int(
            os.getenv("MAX_HISTORY_TURNS", DEFAULT_MAX_HISTORY_TURNS)
        )
        self.summary_threshold = summary_threshold or int(
            os.getenv("MEMORY_SUMMARY_THRESHOLD", DEFAULT_SUMMARY_THRESHOLD)
        )
        self._turns: list[Turn] = []
        self._summary: str = ""

        logger.info(
            f"ConversationMemory initialized — "
            f"max_turns: {self.max_turns} | "
            f"summary_threshold: {self.summary_threshold}"
        )

    def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: Optional[list] = None,
    ) -> None:
        """
        Add a new turn to the conversation history.

        Args:
            role (str): 'user' or 'assistant'.
            content (str): The message content.
            tool_calls (Optional[list]): Tool calls made in this turn.

        Raises:
            ValueError: If role is not 'user' or 'assistant'.
        """
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"Invalid role: '{role}'. Must be 'user' or 'assistant'."
            )
        if not content or not content.strip():
            raise ValueError("Turn content cannot be empty.")

        turn = Turn(
            role=role,
            content=content.strip(),
            tool_calls=tool_calls or [],
        )
        self._turns.append(turn)

        logger.info(
            f"Turn added — role: {role} | "
            f"total turns: {len(self._turns)}"
        )

        # Trigger summarization if threshold is reached
        if len(self._turns) >= self.summary_threshold:
            self._summarize_older_turns()

    def _summarize_older_turns(self) -> None:
        """
        Summarize the oldest turns to free up context window space.

        Keeps the most recent (max_turns - summary_threshold) turns
        in full detail and condenses the rest into a summary string.
        This preserves recent context while compressing older history.
        """
        # Number of turns to keep in full detail
        keep_count = self.max_turns - self.summary_threshold
        keep_count = max(keep_count, 2)  # Always keep at least 2 turns

        turns_to_summarize = self._turns[:-keep_count]
        self._turns = self._turns[-keep_count:]

        if not turns_to_summarize:
            return

        # Build a plain text summary of older turns
        summary_lines = []
        if self._summary:
            summary_lines.append(f"Earlier context: {self._summary}")

        for turn in turns_to_summarize:
            prefix = "User" if turn.role == "user" else "Assistant"
            summary_lines.append(f"{prefix}: {turn.content[:200]}")

        self._summary = " | ".join(summary_lines)

        logger.info(
            f"Summarized {len(turns_to_summarize)} older turn(s) | "
            f"Remaining full turns: {len(self._turns)}"
        )

    def get_context_string(self) -> str:
        """
        Format the conversation history as a string for LLM injection.

        Includes the condensed summary of older turns (if any) followed
        by the full text of recent turns. This is injected into the
        agent's prompt so it has awareness of the conversation so far.

        Returns:
            str: Formatted conversation history string.
                 Returns empty string if no history exists.
        """
        if not self._turns and not self._summary:
            return ""

        parts = []

        if self._summary:
            parts.append(f"[Earlier conversation summary]\n{self._summary}")

        if self._turns:
            parts.append("[Recent conversation]")
            for turn in self._turns:
                prefix = "User" if turn.role == "user" else "Assistant"
                parts.append(f"{prefix}: {turn.content}")

                # Include tool calls if present
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        parts.append(
                            f"  → Tool called: {tc.get('tool', 'unknown')} | "
                            f"Result: {tc.get('result', '')[:100]}"
                        )

        return "\n".join(parts)

    def get_last_user_query(self) -> Optional[str]:
        """
        Return the most recent user query from the conversation history.

        Returns:
            Optional[str]: Last user query, or None if no user turns exist.
        """
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_turn_count(self) -> int:
        """Return the total number of turns in memory (including summarized)."""
        return len(self._turns)

    def clear(self) -> None:
        """
        Clear all conversation history and the summary.

        Called when starting a new conversation session.
        """
        self._turns = []
        self._summary = ""
        logger.info("Conversation memory cleared")

    def to_dict(self) -> dict:
        """
        Serialize memory state to a dict for logging and debugging.

        Returns:
            dict: Memory state including turns and summary.
        """
        return {
            "turn_count": len(self._turns),
            "has_summary": bool(self._summary),
            "summary_preview": self._summary[:200] if self._summary else "",
            "turns": [
                {
                    "role": t.role,
                    "content_preview": t.content[:100],
                    "timestamp": t.timestamp,
                    "tool_calls": t.tool_calls,
                }
                for t in self._turns
            ],
        }


# -------------------------------------------------------------------
# Quick test
# python -m src.memory.memory
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Memory Test ---\n")

    memory = ConversationMemory(max_turns=6, summary_threshold=4)

    turns = [
        ("user", "What is the PTO policy for new employees?"),
        ("assistant", "New employees receive 15 days of PTO in their first year."),
        ("user", "When can I start using my PTO?"),
        ("assistant", "You can use accrued PTO after completing 90 days of service."),
        ("user", "How many days can I carry over?"),
        ("assistant", "You can carry over a maximum of 10 unused PTO days."),
        ("user", "What about sick leave?"),
        ("assistant", "You receive 7 dedicated sick days per calendar year."),
    ]

    for role, content in turns:
        memory.add_turn(role=role, content=content)
        print(f"Added turn [{role}]: {content[:60]}...")

    print(f"\nTotal turns in memory : {memory.get_turn_count()}")
    print(f"Last user query       : {memory.get_last_user_query()}")
    print(f"\n--- Context String ---\n")
    print(memory.get_context_string())
    print(f"\n--- Memory State ---\n")
    import json
    print(json.dumps(memory.to_dict(), indent=2))
