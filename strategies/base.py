"""
Base class for all planning strategies.
Provides a common interface, LLM client setup, and retry logic.
Supports Gemini, OpenAI, and Groq via the OpenAI-compatible API.
"""
import time
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from openai import OpenAI

import config
from tools.agent_tools import get_all_tools


@dataclass
class StrategyResult:
    """Result of a strategy execution."""
    strategy_name: str
    task_id: str
    task_description: str
    final_answer: str
    correct: bool
    expected_answer: str
    execution_time: float          # seconds
    step_count: int
    reasoning_trace: list          # list of (step_type, content) tuples
    tool_calls: int
    failure_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "final_answer": self.final_answer,
            "correct": self.correct,
            "expected_answer": self.expected_answer,
            "execution_time": round(self.execution_time, 3),
            "step_count": self.step_count,
            "tool_calls": self.tool_calls,
            "failure_type": self.failure_type,
            "error_message": self.error_message,
        }


class BaseStrategy:
    """Abstract base class for planning strategies."""

    name: str = "base"

    def __init__(self):
        # Get provider config and create appropriate client
        provider_cfg = config.get_provider_config()
        client_kwargs = {"api_key": provider_cfg["api_key"]}
        if provider_cfg["base_url"]:
            client_kwargs["base_url"] = provider_cfg["base_url"]

        self.client = OpenAI(**client_kwargs)
        self.model = provider_cfg["model"]
        self.provider = config.LLM_PROVIDER
        self.tools = get_all_tools()
        self.max_steps = config.MAX_STEPS_PER_TASK

    def call_llm(self, messages: list, temperature: float = None) -> str:
        """
        Send messages to the LLM and return the response text.
        Includes automatic retry with exponential backoff for rate limits.
        """
        last_error = None
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < config.MAX_RETRIES:
                        wait_time = config.RETRY_BASE_DELAY * (config.RETRY_BACKOFF_FACTOR ** attempt)
                        print(f"      [Rate limit hit — waiting {wait_time}s before retry {attempt+1}/{config.MAX_RETRIES}]")
                        time.sleep(wait_time)
                        continue
                # Non-rate-limit error or final retry — return error
                return f"[LLM_ERROR]: {error_str}"

        return f"[LLM_ERROR]: Max retries exceeded. Last error: {str(last_error)}"

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name with the given input."""
        tool_name_clean = tool_name.strip().lower()
        if tool_name_clean in self.tools:
            try:
                return self.tools[tool_name_clean]["function"](tool_input)
            except Exception as e:
                return f"Tool execution error: {str(e)}"
        return f"Unknown tool: {tool_name}. Available tools: {', '.join(self.tools.keys())}"

    def get_tools_description(self) -> str:
        """Return a formatted description of all available tools."""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def check_answer(self, final_answer: str, expected: str) -> bool:
        """Check if the agent's answer matches the expected answer."""
        if not final_answer or not expected:
            return False
        fa = final_answer.lower().strip().rstrip(".")
        ea = expected.lower().strip().rstrip(".")
        if fa == ea:
            return True
        if ea in fa or fa in ea:
            return True
        try:
            fa_num = float("".join(c for c in fa if c.isdigit() or c in ".-"))
            ea_num = float("".join(c for c in ea if c.isdigit() or c in ".-"))
            return abs(fa_num - ea_num) < 0.01 * max(abs(ea_num), 1)
        except (ValueError, ZeroDivisionError):
            pass
        ea_words = set(ea.split())
        fa_words = set(fa.split())
        if len(ea_words) > 0 and len(ea_words & fa_words) / len(ea_words) > 0.7:
            return True
        return False

    def classify_failure(self, result: "StrategyResult", trace: list) -> str:
        """Classify the type of failure based on the execution trace."""
        if result.error_message and "timeout" in result.error_message.lower():
            return "timeout"
        if result.error_message and ("rate_limit" in result.error_message.lower() or "429" in result.error_message):
            return "rate_limit"
        if result.step_count >= self.max_steps:
            return "loop"
        if result.correct:
            return None
        step_contents = [str(s) for s in trace]
        if len(step_contents) > 3:
            last_3 = step_contents[-3:]
            if len(set(last_3)) == 1:
                return "loop"
        if not result.final_answer or result.final_answer.strip() == "":
            return "incomplete"
        if "[LLM_ERROR]" in result.final_answer:
            return "api_error"
        return "wrong_answer"

    def run(self, task: dict) -> StrategyResult:
        """Execute the strategy on a task. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
