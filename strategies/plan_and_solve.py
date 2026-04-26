"""
Plan-and-Solve Strategy.

Based on: Wang et al., 2023 - "Plan-and-Solve Prompting"

Two distinct phases:
1. PLAN phase: Decompose the task into a sequence of sub-steps
2. SOLVE phase: Execute each sub-step sequentially, using tools as needed
"""
import time
import re
from strategies.base import BaseStrategy, StrategyResult


class PlanAndSolveStrategy(BaseStrategy):
    name = "Plan-and-Solve"

    PLAN_PROMPT = """You are a planning agent. Your job is to decompose a task into clear, sequential sub-steps.

Available tools:
{tools}

Given the task below, create a step-by-step plan. Each step should be a concrete action.
If a step needs a tool, indicate which tool to use.

Respond in this EXACT format:
Plan:
Step 1: <description> [Tool: <tool_name> if needed, else "none"]
Step 2: <description> [Tool: <tool_name> if needed, else "none"]
...
Step N: Combine results and provide the final answer [Tool: none]

Task: {task}
"""

    SOLVE_PROMPT = """You are an execution agent. You are given a plan and must execute one step at a time.

Available tools:
{tools}

Current task: {task}
Plan: {plan}

Previous results:
{previous_results}

Now execute Step {step_num}: {step_description}

If this step requires a tool, respond with:
Action: <tool_name>[<input>]

If this step is reasoning only, respond with:
Reasoning: <your reasoning and result for this step>

If this is the final step, respond with:
Final Answer: <your answer>
"""

    def parse_plan(self, plan_text: str) -> list:
        steps = []
        pattern = r"Step\s*(\d+):\s*(.+?)(?:\[Tool:\s*(\w+)\])?$"
        for line in plan_text.split("\n"):
            line = line.strip()
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                step_num = int(match.group(1))
                description = match.group(2).strip().rstrip("[")
                tool = match.group(3).strip().lower() if match.group(3) else "none"
                if tool == "none":
                    tool = None
                steps.append({"step_num": step_num, "description": description, "tool": tool})
        return steps

    def parse_solve_response(self, response: str) -> dict:
        result = {"action": None, "action_input": None, "reasoning": None, "final_answer": None}

        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL | re.MULTILINE)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", response, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()
            result["action_input"] = action_match.group(2).strip()
            return result

        reasoning_match = re.search(r"Reasoning:\s*(.+?)$", response, re.DOTALL | re.MULTILINE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
            return result

        result["reasoning"] = response.strip()
        return result

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        trace = []
        tool_calls = 0
        step_count = 0
        final_answer = ""
        error_message = None
        tools_desc = self.get_tools_description()

        # === PHASE 1: PLANNING ===
        plan_prompt = self.PLAN_PROMPT.format(tools=tools_desc, task=task["description"])
        plan_response = self.call_llm([
            {"role": "system", "content": "You are a task planning agent."},
            {"role": "user", "content": plan_prompt},
        ])
        step_count += 1
        trace.append(("plan", plan_response))

        if "[LLM_ERROR]" in plan_response:
            error_message = plan_response
            execution_time = time.time() - start_time
            result = StrategyResult(
                strategy_name=self.name, task_id=task["id"],
                task_description=task["description"], final_answer="",
                correct=False, expected_answer=task.get("expected_answer", ""),
                execution_time=execution_time, step_count=step_count,
                reasoning_trace=trace, tool_calls=0, error_message=error_message,
            )
            result.failure_type = "api_error"
            return result

        steps = self.parse_plan(plan_response)
        if not steps:
            steps = [
                {"step_num": 1, "description": "Research and gather information", "tool": "search_knowledge"},
                {"step_num": 2, "description": "Provide the final answer", "tool": None},
            ]
        trace.append(("parsed_plan", [s["description"] for s in steps]))

        # === PHASE 2: SOLVING ===
        previous_results = []
        for step_info in steps:
            if step_count >= self.max_steps:
                break
            step_count += 1
            prev_text = "\n".join(
                [f"Step {i+1} result: {r}" for i, r in enumerate(previous_results)]
            ) if previous_results else "None yet."

            solve_prompt = self.SOLVE_PROMPT.format(
                tools=tools_desc, task=task["description"], plan=plan_response,
                previous_results=prev_text, step_num=step_info["step_num"],
                step_description=step_info["description"],
            )
            response = self.call_llm([
                {"role": "system", "content": "You are a task execution agent. Execute the current step precisely."},
                {"role": "user", "content": solve_prompt},
            ])
            trace.append(("solve_step", f"Step {step_info['step_num']}: {response}"))

            if "[LLM_ERROR]" in response:
                error_message = response
                break

            parsed = self.parse_solve_response(response)
            if parsed["final_answer"]:
                final_answer = parsed["final_answer"]
                trace.append(("final_answer", final_answer))
                break
            if parsed["action"]:
                observation = self.execute_tool(parsed["action"], parsed["action_input"])
                tool_calls += 1
                trace.append(("action", f"{parsed['action']}[{parsed['action_input']}]"))
                trace.append(("observation", observation))
                previous_results.append(f"{parsed['action']}: {observation}")
            elif parsed["reasoning"]:
                previous_results.append(parsed["reasoning"])

        if not final_answer and not error_message:
            step_count += 1
            prev_text = "\n".join([f"Step {i+1}: {r}" for i, r in enumerate(previous_results)])
            wrap_up = self.call_llm([
                {"role": "system", "content": "You are a task execution agent."},
                {"role": "user", "content": (
                    f"Task: {task['description']}\n\nYou have completed all steps. Results:\n{prev_text}\n\n"
                    f"Now provide your Final Answer based on these results. Be concise.\nFinal Answer:"
                )},
            ])
            final_answer = wrap_up.replace("Final Answer:", "").strip()
            trace.append(("final_answer", final_answer))

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
