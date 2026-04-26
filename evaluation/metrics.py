"""
Evaluation metrics for comparing planning strategies.
"""
import pandas as pd
import numpy as np
from typing import List
from strategies.base import StrategyResult


def compute_metrics(results: List[StrategyResult]) -> pd.DataFrame:
    records = [r.to_dict() for r in results]
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    metrics = df.groupby("strategy").agg(
        total_tasks=("task_id", "count"),
        tasks_correct=("correct", "sum"),
        success_rate=("correct", "mean"),
        avg_execution_time=("execution_time", "mean"),
        std_execution_time=("execution_time", "std"),
        avg_step_count=("step_count", "mean"),
        std_step_count=("step_count", "std"),
        avg_tool_calls=("tool_calls", "mean"),
        total_tool_calls=("tool_calls", "sum"),
    ).reset_index()
    metrics["success_rate"] = (metrics["success_rate"] * 100).round(1)
    metrics["avg_execution_time"] = metrics["avg_execution_time"].round(3)
    metrics["avg_step_count"] = metrics["avg_step_count"].round(1)
    return metrics


def compute_metrics_by_complexity(results: List[StrategyResult]) -> pd.DataFrame:
    records = []
    for r in results:
        d = r.to_dict()
        if r.task_id.startswith("low"): d["complexity"] = "low"
        elif r.task_id.startswith("med"): d["complexity"] = "medium"
        elif r.task_id.startswith("high"): d["complexity"] = "high"
        else: d["complexity"] = "unknown"
        records.append(d)
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    metrics = df.groupby(["strategy", "complexity"]).agg(
        total_tasks=("task_id", "count"),
        tasks_correct=("correct", "sum"),
        success_rate=("correct", "mean"),
        avg_execution_time=("execution_time", "mean"),
        avg_step_count=("step_count", "mean"),
        avg_tool_calls=("tool_calls", "mean"),
    ).reset_index()
    metrics["success_rate"] = (metrics["success_rate"] * 100).round(1)
    return metrics


def compute_failure_analysis(results: List[StrategyResult]) -> pd.DataFrame:
    records = [r.to_dict() for r in results if not r.correct]
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["strategy", "failure_type", "count"])
    return df.groupby(["strategy", "failure_type"]).size().reset_index(name="count")


def compute_stability_metrics(all_run_results: dict) -> pd.DataFrame:
    records = []
    for run_num, results in all_run_results.items():
        for r in results:
            d = r.to_dict()
            d["run"] = run_num
            records.append(d)
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    stability = df.groupby(["strategy", "task_id"]).agg(
        success_count=("correct", "sum"),
        total_runs=("correct", "count"),
        mean_time=("execution_time", "mean"),
        std_time=("execution_time", "std"),
        mean_steps=("step_count", "mean"),
        std_steps=("step_count", "std"),
    ).reset_index()
    stability["consistency"] = (stability["success_count"] / stability["total_runs"] * 100).round(1)
    stability["time_cv"] = (stability["std_time"] / stability["mean_time"]).round(3)
    return stability
