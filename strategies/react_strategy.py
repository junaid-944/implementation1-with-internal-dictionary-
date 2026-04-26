"""
ReAct Strategy: Synergizing Reasoning and Acting.

Based on: Yao et al., 2023 - "ReAct: Synergizing Reasoning and Acting in Language Models"

The agent interleaves Thought, Action, and Observation steps:
1. Thought: Reason about the current state and what to do next
2. Action: Choose and execute a tool or provide a final answer
3. Observation: Observe the result and feed it back
"""
import time
import re
from strategies.base import BaseStrategy, StrategyResult


class ReActStrategy(BaseStrategy):
    name = "ReAct"

    SYSTEM_PROMPT = """You are an AI agent that solves tasks using the ReAct framework.
You must interleave Thought, Action, and Observation steps.

Available tools:
{tools}

Response format — follow this EXACTLY for each step:

Thought: <your reasoning about what to do next>
Action: <tool_name>[<input>]

When you have the final answer, respond with:
Thought: <your final reasoning>
Final Answer: <your answer>

Rules:
- Always start with a Thought.
- Use exactly one Action per step (or give a Final Answer).
- Do NOT make up information. Use tools to find facts.
- Be concise and precise in your final answer.
- You MUST use tools for any factual lookups or calculations. Do NOT answer from memory.
"""

    def parse_response(self, response: str) -> dict:
        result = {"thought": None, "action": None, "action_input": None, "final_answer": None}

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL | re.MULTILINE)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", response, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()
            result["action_input"] = action_match.group(2).strip()

        return result

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        trace = []
        tool_calls = 0
        step_count = 0
        final_answer = ""
        error_message = None

        system_msg = self.SYSTEM_PROMPT.format(tools=self.get_tools_description())
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Task: {task['description']}"},
        ]

        try:
            for step in range(self.max_steps):
                step_count += 1
                response = self.call_llm(messages)
                trace.append(("llm_response", response))

                if "[LLM_ERROR]" in response:
                    error_message = response
                    break

                parsed = self.parse_response(response)

                if parsed["thought"]:
                    trace.append(("thought", parsed["thought"]))

                if parsed["final_answer"]:
                    final_answer = parsed["final_answer"]
                    trace.append(("final_answer", final_answer))
                    break

                if parsed["action"]:
                    observation = self.execute_tool(parsed["action"], parsed["action_input"])
                    tool_calls += 1
                    trace.append(("action", f"{parsed['action']}[{parsed['action_input']}]"))
                    trace.append(("observation", observation))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": "Please continue with an Action or provide a Final Answer."})

        except Exception as e:
            error_message = str(e)

        execution_time = time.time() - start_time
        correct = self.check_answer(final_answer, task.get("expected_answer", ""))

        result = StrategyResult(
            strategy_name=self.name, task_id=task["id"],
            task_description=task["description"], final_answer=final_answer,
            correct=correct, expected_answer=task.get("expected_answer", ""),
            execution_time=execution_time, step_count=step_count,
            reasoning_trace=trace, tool_calls=tool_calls, error_message=error_message,
        )
        result.failure_type = self.classify_failure(result, trace)
        return result
