"""
GEPA-Kontex Integration Package

Integra o GEPA (Genetic Pareto Prompt Optimizer) com o Kontex para
otimização de prompts de aquisição de conhecimento tácito.

Uso básico
----------
    from kontex_gepa import (
        EnvConfig,
        KontexFlowGeneralized,
        GEvalMetric,
        generate_hotpot_pareto_dataset,
    )
"""

from kontex_gepa import (
    AverageDiffScore,
    AzureOpenAI,
    EnvConfig,
    GEVAL_CRITERIA_HOTPOTQA,
    GEVAL_CRITERIA_TABLE,
    GEvalMetric,
    KontexFlow,
    KontexFlowGeneralized,
    LLMJudgeMetric,
    generate_hotpot_pareto_dataset,
    generate_pareto_dataset,
    get_data_from_df,
    parse_users_info_to_specialists,
    run_conversation_simulation,
)

__version__ = "0.1.0"

__all__ = [
    "EnvConfig",
    "AzureOpenAI",
    "run_conversation_simulation",
    "KontexFlow",
    "KontexFlowGeneralized",
    "AverageDiffScore",
    "GEVAL_CRITERIA_HOTPOTQA",
    "GEVAL_CRITERIA_TABLE",
    "GEvalMetric",
    "LLMJudgeMetric",
    "generate_pareto_dataset",
    "generate_hotpot_pareto_dataset",
    "get_data_from_df",
    "parse_users_info_to_specialists",
]
