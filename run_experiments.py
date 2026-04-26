"""
Main experiment runner for the Agentic Planning Study.

Usage:
    python run_experiments.py                     # Run all tasks with all strategies
    python run_experiments.py --complexity low     # Run only low-complexity tasks
    python run_experiments.py --strategy ReAct     # Run only ReAct strategy
    python run_experiments.py --runs 5             # Run 5 times for stability analysis
    python run_experiments.py --quick              # Quick test: 1 task per complexity, 1 run
"""
import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from strategies import ReActStrategy, PlanAndSolveStrategy, TreeOfThoughtsStrategy
from tasks import get_all_tasks, get_tasks_by_complexity
from evaluation.metrics import compute_metrics, compute_metrics_by_complexity, compute_failure_analysis, compute_stability_metrics
from evaluation.visualizer import ResultVisualizer


STRATEGIES = {
    "ReAct": ReActStrategy,
    "Plan-and-Solve": PlanAndSolveStrategy,
    "Tree-of-Thoughts": TreeOfThoughtsStrategy,
}


def run_single_experiment(strategy_name, strategy_class, tasks, verbose=True):
    strategy = strategy_class()
    results = []
    for i, task in enumerate(tasks):
        if verbose:
            print(f"  [{i+1}/{len(tasks)}] Task {task['id']}: {task['description'][:60]}...")
        result = strategy.run(task)
        results.append(result)
        if verbose:
            status = "\u2713" if result.correct else "\u2717"
            print(f"         {status} Answer: {result.final_answer[:80] if result.final_answer else 'N/A'}")
            print(f"           Time: {result.execution_time:.2f}s | Steps: {result.step_count} | Tools: {result.tool_calls}")
    return results


def run_full_experiment(strategies_to_run, tasks, num_runs=1, verbose=True):
    all_results = []
    all_run_results = {}
    for run in range(1, num_runs + 1):
        if num_runs > 1:
            print(f"\n{'='*60}\nRUN {run}/{num_runs}\n{'='*60}")
        run_results = []
        for strategy_name in strategies_to_run:
            strategy_class = STRATEGIES[strategy_name]
            print(f"\n--- Strategy: {strategy_name} ---")
            results = run_single_experiment(strategy_name, strategy_class, tasks, verbose)
            run_results.extend(results)
        all_results.extend(run_results)
        all_run_results[run] = run_results
    return all_results, all_run_results


def print_summary(results):
    metrics = compute_metrics(results)
    complexity_metrics = compute_metrics_by_complexity(results)
    failures = compute_failure_analysis(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Success%':>10} {'Avg Time':>10} {'Avg Steps':>10} {'Tool Calls':>12}")
    print("-" * 70)
    for _, row in metrics.iterrows():
        print(f"{row['strategy']:<20} {row['success_rate']:>9.1f}% {row['avg_execution_time']:>9.2f}s {row['avg_step_count']:>10.1f} {row['total_tool_calls']:>12}")

    print(f"\n{'Strategy':<20} {'Complexity':<10} {'Success%':>10} {'Avg Time':>10} {'Avg Steps':>10}")
    print("-" * 70)
    for _, row in complexity_metrics.iterrows():
        print(f"{row['strategy']:<20} {row['complexity']:<10} {row['success_rate']:>9.1f}% {row['avg_execution_time']:>9.2f}s {row['avg_step_count']:>10.1f}")

    if not failures.empty:
        print("\nFailure Analysis:")
        print("-" * 50)
        for _, row in failures.iterrows():
            print(f"  {row['strategy']}: {row['failure_type']} ({row['count']})")


def main():
    parser = argparse.ArgumentParser(description="Run the Agentic Planning Study experiments.")
    parser.add_argument("--complexity", choices=["low", "medium", "high"])
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Validate provider config
    try:
        provider_cfg = config.get_provider_config()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nSetup:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API key for the provider you want to use")
        print("  3. Set LLM_PROVIDER=gemini (or openai, or groq)")
        sys.exit(1)

    if args.complexity:
        tasks = get_tasks_by_complexity(args.complexity)
    else:
        tasks = get_all_tasks()

    if args.quick:
        quick_tasks = []
        for c in ["low", "medium", "high"]:
            ct = get_tasks_by_complexity(c)
            if ct: quick_tasks.append(ct[0])
        tasks = quick_tasks

    strategies_to_run = [args.strategy] if args.strategy else list(STRATEGIES.keys())

    print(f"Agentic Planning Strategy Comparison Study")
    print(f"   Provider: {config.LLM_PROVIDER}")
    print(f"   Model: {provider_cfg['model']}")
    print(f"   Strategies: {', '.join(strategies_to_run)}")
    print(f"   Tasks: {len(tasks)} (Runs: {args.runs})")
    print(f"   Retry: up to {config.MAX_RETRIES} retries with {config.RETRY_BASE_DELAY}s base delay")
    print(f"   Output: {config.RESULTS_DIR}")

    start = time.time()
    all_results, all_run_results = run_full_experiment(
        strategies_to_run, tasks, num_runs=args.runs, verbose=args.verbose
    )
    total_time = time.time() - start

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print_summary(all_results)

    if not args.no_viz:
        viz = ResultVisualizer(all_results, all_run_results)
        viz.generate_all()

    raw_path = os.path.join(config.RESULTS_DIR, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\nRaw results saved to: {raw_path}")

    if args.runs > 1:
        stability = compute_stability_metrics(all_run_results)
        stability.to_csv(os.path.join(config.RESULTS_DIR, "stability_analysis.csv"), index=False)
        agg = stability.groupby("strategy").agg(
            avg_consistency=("consistency", "mean"),
            avg_time_cv=("time_cv", "mean"),
        ).reset_index()
        print("\nStability Summary:")
        for _, row in agg.iterrows():
            print(f"  {row['strategy']}: {row['avg_consistency']:.1f}% consistent, time CV={row['avg_time_cv']:.3f}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
