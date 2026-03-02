from __future__ import annotations

from crewai import Agent


def build_planner_agent(llm):
    return Agent(
        role="PlannerAgent",
        goal="Design the best trial-block plan for NAS using the given constraints.",
        backstory=(
            "You are a meticulous experiment planner. You optimize tradeoffs between "
            "search breadth and execution cost while respecting exact trial budgets."
        ),
        verbose=True,
        llm=llm,
    )


def build_runner_agent(llm, tools):
    return Agent(
        role="RunnerAgent",
        goal="Execute each planned NAS block precisely by using provided tools.",
        backstory="You run experiments safely and return concise execution status.",
        tools=tools,
        verbose=True,
        llm=llm,
    )


def build_reporter_agent(llm, tools):
    return Agent(
        role="ReporterAgent",
        goal="Summarize outcomes and point to generated artifacts.",
        backstory="You synthesize experiment outputs into a compact operational report.",
        tools=tools,
        verbose=True,
        llm=llm,
    )
