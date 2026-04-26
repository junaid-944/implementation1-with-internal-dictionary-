"""
Tree of Thoughts (ToT) Strategy.

Based on: Yao et al., 2023 - "Tree of Thoughts: Deliberate Problem Solving
with Large Language Models"

Explores multiple reasoning paths (branches) at each step, evaluates them,
and selects the most promising path to continue.
"""
import time
import re
from strategies.base import BaseStrategy, StrategyResult
import config


class TreeOfThoughtsStrategy(BaseStrategy):
    name = "Tree-of-Thoughts"

    GENERATE_PROMPT = """You are a reasoning agent solving a task step by step.

Available tools:
{tools}

Task: {task}

{context}

Generate ONE possible next thought/action for solving this task.
Approach it from perspective #{perspective_num}.

If you need to use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>[<input>]

If you can provide a final answer based on available information, respond with:
Thought: <your reasoning>
Final Answer: <your answer>

If you need to reason further without a tool, respond with:
Thought: <your reasoning>
Continue: <summary of what you've figured out so far>
"""

    EVALUATE_PROMPT = """You are evaluating different reasoning approaches for a task.

Task: {task}

Here are {num_candidates} candidate approaches:

{candidates}

Evaluate each candidate on a scale of 1-10 based on:
- Correctness of reasoning (is the logic sound?)
- Progress toward the answer (how close is it to solving the task?)
- Efficiency (is it taking a reasonable approach?)

Respond with ONLY the candidate number (1-{num_candidates}) that is best:
Best candidate: <number>
"""

    def parse_generate_response(self, response: str) -> dict:
        result = {"thought": None, "action": None, "action_input": None,
                  "final_answer": None, "continue_summary": None}

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\nContinue:|\Z)",
                                  response, re.DOTALL)
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

        continue_match = re.search(r"Continue:\s*(.+?)$", response, re.DOTALL | re.MULTILINE)
        if continue_match:
            result["continue_summary"] = continue_match.group(1).strip()

        return result

    def generate_candidates(self, task, context, num_branches):
        candidates = []
        tools_desc = self.get_tools_description()
        for i in range(num_branches):
            prompt = self.GENERATE_PROMPT.format(
                tools=tools_desc, task=task["description"],
                context=context if context else "No previous steps yet.",
                perspective_num=i + 1,
            )
            response = self.call_llm(
                [{"role": "system", "content": f"You are reasoning agent #{i+1}. Think creatively."},
                 {"role": "user", "content": prompt}],
                temperature=0.7,
            )
            parsed = self.parse_generate_response(response)
            candidates.append({"raw": response, "parsed": parsed, "index": i + 1})
        return candidates

    def evaluate_candidates(self, task, candidates):
        if len(candidates) == 1:
            return 0
        candidate_text = ""
        for i, c in enumerate(candidates):
            candidate_text += f"\nCandidate {i+1}:\n{c['raw']}\n"
        prompt = self.EVALUATE_PROMPT.format(
            task=task["description"], num_candidates=len(candidates), candidates=candidate_text,
        )
        response = self.call_llm([
            {"role": "system", "content": "You are an impartial evaluator of reasoning quality."},
            {"role": "user", "content": prompt},
        ])
        match = re.search(r"Best candidate:\s*(\d+)", response)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return idx
        return 0

    def run(self, task: dict) -> StrategyResult:
        start_time = time.time()
        trace = []
        tool_calls = 0
        step_count = 0
        final_answer = ""
        error_message = None
        context_parts = []
        num_branches = config.TOT_BRANCHES
        max_depth = config.TOT_MAX_DEPTH

        try:
            for depth in range(max_depth):
                if step_count >= self.max_steps:
                    break
                context = "\n".join(context_parts) if context_parts else ""

                step_count += 1
                candidates = self.generate_candidates(task, context, num_branches)
                trace.append(("generate", f"Depth {depth+1}: Generated {len(candidates)} candidates"))

                final_candidates = [c for c in candidates if c["parsed"]["final_answer"]]
                if final_candidates:
                    if len(final_candidates) > 1:
                        step_count += 1
                        best_idx = self.evaluate_candidates(task, final_candidates)
                        best = final_candidates[best_idx]
                    else:
                        best = final_candidates[0]
                    final_answer = best["parsed"]["final_answer"]
                    trace.append(("final_answer", final_answer))
                    break

                step_count += 1
                best_idx = self.evaluate_candidates(task, candidates)
                best = candidates[best_idx]
                trace.append(("evaluate", f"Selected candidate {best_idx + 1}"))
                trace.append(("best_thought", best["parsed"].get("thought", "")))

                if best["parsed"]["action"]:
                    observation = self.execute_tool(best["parsed"]["action"], best["parsed"]["action_input"])
                    tool_calls += 1
                    trace.append(("action", f"{best['parsed']['action']}[{best['parsed']['action_input']}]"))
                    trace.append(("observation", observation))
                    context_parts.append(
                        f"Step {depth+1}: {best['parsed']['thought']}\n"
                        f"Action: {best['parsed']['action']}[{best['parsed']['action_input']}]\n"
                        f"Observation: {observation}"
                    )
                elif best["parsed"]["continue_summary"]:
                    context_parts.append(
                        f"Step {depth+1}: {best['parsed']['thought']}\n"
                        f"Progress: {best['parsed']['continue_summary']}"
                    )
                else:
                    context_parts.append(f"Step {depth+1}: {best['raw']}")

            if not final_answer and not error_message:
                step_count += 1
                context = "\n".join(context_parts)
                wrap_up = self.call_llm([
                    {"role": "system", "content": "You are a reasoning agent. Provide a final answer."},
                    {"role": "user", "content": (
                        f"Task: {task['description']}\n\nYour reasoning so far:\n{context}\n\n"
                        f"Based on all the above, provide your Final Answer. Be concise and precise.\nFinal Answer:"
                    )},
                ])
                final_answer = wrap_up.replace("Final Answer:", "").strip()
                trace.append(("final_answer", final_answer))

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
            metadata={"branches": num_branches, "max_depth": max_depth},
        )
        result.failure_type = self.classify_failure(result, trace)
        return result
