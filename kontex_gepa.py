"""
kontex_gepa.py

Integração do GEPA (Genetic Pareto Prompt Optimizer) com o Kontex para
otimização de prompts de aquisição de conhecimento tácito.

Exporta:
    - EnvConfig: carrega variáveis de ambiente (.env)
    - AzureOpenAI: wrapper DeepEval para Azure OpenAI
    - run_conversation_simulation: executa uma simulação de conversa Kontex
    - KontexFlow: fluxo de controle para o workflow de mineração (mining/EDD)
    - KontexFlowGeneralized: fluxo de controle para o workflow HotPotQA
    - AverageDiffScore, GEvalMetric, LLMJudgeMetric: métricas de avaliação
    - generate_pareto_dataset: gera dataset via EDD (mining simulado)
    - generate_hotpot_pareto_dataset: gera dataset via HotPotQA
    - get_data_from_df: carrega dataset a partir de um DataFrame CSV
    - parse_users_info_to_specialists: converte string CSV em Specialists
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer, util

from deepeval.metrics import GEval, BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ---------------------------------------------------------------------------
# Configuração de paths — permite executar tanto como script quanto como módulo
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).parent
_parent_dir = _this_dir.parent

sys.path.insert(0, str(_parent_dir))
sys.path.insert(0, str(_parent_dir / "kontex" / "src"))
sys.path.insert(0, str(_parent_dir / "gepa" / "src"))

# Banco de dados persistente (deve ser definido antes de importar módulos kontex)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_this_dir}/kontex_gepa_data.db")

# ---------------------------------------------------------------------------
# Imports Kontex & GEPA
# ---------------------------------------------------------------------------
from kontex.database import db
from kontex.knowledge import CollectedKnowledge
from kontex.logging import logger
from kontex.orquestration import ConversationalWrapper
from kontex.settings import settings
from kontex.simulation.edd.edd_run_params import EDDRunConfig
from kontex.simulation.edd.general_knowledge import DomainKnowledge, FullKnowledge
from kontex.simulation.edd.simulation import edd_simulation
from kontex.llm.agents import DummyAgent
from kontex.llm.scheduler import LLMScheduler
from kontex.specialist import Specialist
from kontex.llm.agents.hotpotqa_agents import (
    hotpotqa_description_building_role,
    hotpotqa_questioning_role,
    hotpotqa_self_critique_role,
    hotpotqa_subject_change_role,
)

from gepa import GEPAConfig, GEPAOptimizer
from gepa.config import (
    DatabaseConfig,
    InferenceConfig,
    ObservabilityConfig,
    OptimizationConfig,
)
from gepa.core.system import CompoundAISystem, IOSchema, LanguageModule, SequentialFlow
from gepa.evaluation.base import (
    Evaluator,
    EvaluationResult,
    SimpleFeedbackEvaluator,
    SimpleEvaluator,
)
from gepa.evaluation.metrics import ExactMatch, F1Score, Metric
from gepa.inference.factory import InferenceFactory

from parse_full_knowledge import parse_full_knowledge_from_string

# ---------------------------------------------------------------------------
# Helpers de ambiente e LLM
# ---------------------------------------------------------------------------

class EnvConfig:
    """Carrega variáveis de ambiente a partir de um arquivo .env."""

    def __init__(self, env_file: str = ".env"):
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Variáveis de ambiente carregadas de {env_file}")
        else:
            print(f"⚠ Arquivo {env_file} não encontrado")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_MODEL")


class AzureOpenAI(DeepEvalBaseLLM):
    """Wrapper DeepEval para um modelo AzureChatOpenAI (LangChain)."""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return "Custom Azure OpenAI Model"


# ---------------------------------------------------------------------------
# Simulação de conversa Kontex
# ---------------------------------------------------------------------------

def run_conversation_simulation(
    run_id: UUID,
    simulated_users: dict[str, Specialist],
    full_knowledge: FullKnowledge,
    prompts: dict[str, str] | None = None,
    seed: int | None = None,
    change_roles: bool = False,
    new_roles: dict[str, str] | None = None,
    external_question: str | None = None,
) -> tuple[dict[str, str], float]:
    """
    Executa uma simulação de conversa Kontex para cada domínio em full_knowledge.

    Parameters
    ----------
    run_id:
        UUID da execução (usado para rastreamento no banco de dados).
    simulated_users:
        Dicionário {nome: Specialist} dos usuários simulados.
    full_knowledge:
        Objeto FullKnowledge com os domínios e fatos do conhecimento.
    prompts:
        Dicionário de prompts de usuário para cada papel. None usa os
        templates padrão internos do agente.
    seed:
        Semente para reproducibilidade na escolha do usuário inicial.
    change_roles:
        Se True, usa new_roles para sobrescrever os system prompts dos agentes.
    new_roles:
        Dicionário {papel: system_prompt} para sobrescrever os papéis dos agentes.
    external_question:
        Questão externa a ser respondida pela simulação (ex.: HotPotQA).

    Returns
    -------
    tuple[dict[str, str], float]
        (descriptions, final_critique_score) onde descriptions mapeia
        table_name -> description produzida pela simulação.
    """
    rng = random.Random(seed)
    descriptions: dict[str, str] = {}

    for table_name, table_knowledge in full_knowledge.domains.items():
        table_columns = list(table_knowledge.facts.keys())
        initial_description = f"Table: {table_name}\nColumns: {table_columns}"
        table = CollectedKnowledge(table_name, initial_description)

        scheduler = (
            LLMScheduler(maxhist=0, new_roles=new_roles)
            if change_roles
            else LLMScheduler(maxhist=0)
        )

        conversational_wrapper = ConversationalWrapper(
            scheduler,
            prompts,
            simulated_users,
            run_id,
        )
        initial_user = rng.choice(list(simulated_users.keys()))
        description, final_critique_score = conversational_wrapper.run_conversation(
            table,
            initial_user,
            min_description_quality=3,
            max_single_conversation=1,
            max_conversation_depth=10,
            external_question=external_question,
            external_data=True,
            skip_hi_n_bye=True,
        )

        logger.info(f"Final Table Description:\n{description}")
        logger.info(f"\n-------------\nOriginal Description: \n{table_knowledge.facts}")
        logger.info(f"Final Critique Score: {final_critique_score}")
        descriptions[table_name] = description

    return descriptions, final_critique_score


# ---------------------------------------------------------------------------
# Fluxos de controle GEPA
# ---------------------------------------------------------------------------

class KontexFlow:
    """
    Fluxo de controle para o workflow de mineração (dados simulados via EDD).

    O GEPA otimiza o prompt do módulo 'questioning'. O crítico usa um
    prompt fixo embutido neste fluxo.
    """

    async def execute(
        self,
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any,
    ) -> Dict[str, Any]:
        current_data = input_data.copy()

        prompts = {
            "questioning_prompt": modules["questioning"].prompt,
            "critique_prompt": (
                "Evaluate the completeness of this table description:\n\n"
                "{tacit_knowledge}\n\n"
                "Provide assessment in this format:\n"
                "Score: [0-10]\n"
                "Reasoning: [why this score]\n"
                "Suggestions: [what's missing]\n\n"
                "To score high (8+), description needs:\n"
                "- All column names and meanings\n"
                "- Data types for each column\n"
                "- Example values where relevant\n"
                "- Business context and purpose"
            ),
        }

        current_run_id = input_data.get("run_id", UUID(int=0))

        description, final_critique_score = run_conversation_simulation(
            run_id=current_run_id,
            prompts=prompts,
            simulated_users=input_data["users_with_knowledge"],
            full_knowledge=input_data["full_knowledge"],
            seed=42,
        )

        if final_critique_score is None:
            final_critique_score = 0.0
        else:
            try:
                final_critique_score = float(final_critique_score)
            except (ValueError, TypeError):
                final_critique_score = 0.0

        current_data["current_run_id"] = current_run_id
        current_data["description"] = description
        current_data["output"] = final_critique_score
        logger.debug(f"Final critic score: {final_critique_score}")
        return current_data


class KontexFlowGeneralized:
    """
    Fluxo de controle generalizado para o workflow HotPotQA.

    O GEPA otimiza o prompt do módulo 'description_building' (system prompt
    do agente de construção de descrição). Os demais papéis (questioning,
    subject_change, self_critique) usam os templates padrão do HotPotQA.
    """

    async def execute(
        self,
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any,
    ) -> Dict[str, Any]:
        current_data = input_data.copy()

        # O GEPA evolui modules['description_building'].prompt
        new_roles = {
            "questioning": hotpotqa_questioning_role,
            "subject_change": hotpotqa_subject_change_role,
            "self_critique": hotpotqa_self_critique_role,
            "description_building": modules["description_building"].prompt,
        }

        # None → usa o template de mensagem padrão interno de cada agente
        prompts = {
            "questioning": None,
            "self_critique": None,
            "description_building": None,
        }

        raw_run_id = input_data.get("run_id", UUID(int=0))
        current_run_id = getattr(raw_run_id, "id", raw_run_id)

        description, final_critique_score = run_conversation_simulation(
            run_id=current_run_id,
            prompts=prompts,
            simulated_users=input_data.get("users_with_knowledge", {}),
            full_knowledge=input_data.get("full_knowledge"),
            change_roles=True,
            new_roles=new_roles,
            seed=42,
            external_question=input_data.get("question"),
        )

        current_data["current_run_id"] = current_run_id
        current_data["question"] = input_data.get("question", "")

        try:
            current_data["output"] = (
                float(final_critique_score) if final_critique_score is not None else 0.0
            )
        except (ValueError, TypeError):
            logger.warning(
                f"Não foi possível converter score {final_critique_score} para float. "
                "Usando 0.0."
            )
            current_data["output"] = 0.0

        current_data["description"] = description
        logger.debug(f"Final critic score: {current_data['output']}")
        return current_data


# ---------------------------------------------------------------------------
# Métricas de avaliação
# ---------------------------------------------------------------------------

class AverageDiffScore(Metric):
    """Diferença média entre score esperado (10) e score do crítico."""

    def __init__(self, name: str = "score"):
        super().__init__(name)

    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        scores = [10 - (ref - pred["output"]) for pred, ref in zip(predictions, references)]
        return float(np.mean(np.array(scores)))


# ---------------------------------------------------------------------------
# Critérios padrão para GEvalMetric — podem ser sobrescritos nos notebooks
# ---------------------------------------------------------------------------

#: Critérios para o workflow HotPotQA (resposta a perguntas sobre múltiplos documentos)
GEVAL_CRITERIA_HOTPOTQA = {
    "factual_accuracy": (
        "Evaluate whether the answer produced by the questioning agent, based on facts "
        "collected from specialist agents, contains any fabricated, hallucinated, or "
        "incorrect information when compared to the expected answer. Penalize heavily "
        "for facts invented by the agent that are not supported by or contradict the "
        "expected answer."
    ),
    "completeness": (
        "Evaluate how thoroughly the questioning agent collected relevant facts from "
        "specialist agents to answer the original question. Assess whether the produced "
        "answer covers all key information present in the expected answer, and penalize "
        "for important facts or details that are missing."
    ),
}

#: Critérios para o workflow de tabelas (mineração / dados simulados via EDD)
GEVAL_CRITERIA_TABLE = {
    "factual_accuracy": (
        "Evaluate whether the table description produced by the agent accurately reflects "
        "the expected description. Check that column names, data types, and business meanings "
        "are correctly reported. Penalize heavily for invented columns, wrong data types, or "
        "descriptions that contradict the expected reference."
    ),
    "completeness": (
        "Evaluate how completely the agent described the table. Assess whether all columns "
        "are covered with their names, data types, and business context. Penalize for missing "
        "columns, absent data type information, or lack of business meaning for relevant fields."
    ),
}


class GEvalMetric(Metric):
    """
    Métrica composta baseada em GEval (DeepEval).

    Os critérios de avaliação são configuráveis via ``factual_accuracy_criteria``
    e ``completeness_criteria``, permitindo adaptar a métrica ao contexto de uso
    (HotPotQA, tabelas, etc.) sem alterar o código da classe.

    Use as constantes ``GEVAL_CRITERIA_HOTPOTQA`` ou ``GEVAL_CRITERIA_TABLE``
    como ponto de partida, ou defina critérios próprios no início do notebook.

    Exemplo::

        evaluator = SimpleFeedbackEvaluator([
            GEvalMetric(
                factual_accuracy_criteria=GEVAL_CRITERIA_TABLE["factual_accuracy"],
                completeness_criteria=GEVAL_CRITERIA_TABLE["completeness"],
            )
        ])
    """

    def __init__(
        self,
        name: str = "geval_metric",
        run_id: UUID | None = None,
        factual_accuracy_criteria: str | None = None,
        completeness_criteria: str | None = None,
    ):
        super().__init__(name)
        self.run_id = run_id
        self.factual_accuracy_criteria = (
            factual_accuracy_criteria
            or GEVAL_CRITERIA_HOTPOTQA["factual_accuracy"]
        )
        self.completeness_criteria = (
            completeness_criteria
            or GEVAL_CRITERIA_HOTPOTQA["completeness"]
        )

        config = EnvConfig(env_file=str(_this_dir / ".env"))
        self.api_key = config.api_key
        self.model = config.model

        azure_endpoint = "https://azureopenai4k.openai.azure.com/"
        openai_api_version = "2025-01-01-preview"
        azure_deployment = "gpt-5-mini"

        custom_model = AzureChatOpenAI(
            model=self.model,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            openai_api_key=self.api_key,
            openai_api_version=openai_api_version,
        )
        self.azure_openai = AzureOpenAI(model=custom_model)

    def compute(
        self,
        prediction_description: Any,
        reference_description: Any,
    ) -> tuple[float, str]:
        weight_hallucination = 0.6
        weight_completeness = 0.4

        if isinstance(prediction_description, list) and prediction_description:
            pred_data = prediction_description[0]
            if "current_run_id" in pred_data:
                self.run_id = pred_data["current_run_id"]

        reference_description = reference_description[0]["expected_description"]
        question = prediction_description[0].get(
            "question", "Answer the question based on the collected facts."
        )
        raw_desc = prediction_description[0]["description"]
        table_name = list(raw_desc.keys())[0]
        prediction_str = raw_desc[table_name]

        factual_accuracy = GEval(
            name="Factual Accuracy",
            model=self.azure_openai,
            criteria=self.factual_accuracy_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )

        completeness = GEval(
            name="Completeness",
            model=self.azure_openai,
            criteria=self.completeness_criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=prediction_str,
            expected_output=reference_description,
            retrieval_context=[reference_description],
        )

        fa_score, fa_reason = self._convergence_loop(
            factual_accuracy, test_case, n_runs=20, max_retries=3, min_std_error=0.05, n_runs_min=5
        )
        comp_score, comp_reason = self._convergence_loop(
            completeness, test_case, n_runs=20, max_retries=3, min_std_error=0.05, n_runs_min=5
        )

        aggregated_reasoning = self._aggregate_reasons([fa_reason, comp_reason])
        overall_score = weight_hallucination * fa_score + weight_completeness * comp_score

        if self.run_id:
            try:
                db.add_geval_metric(run_id=self.run_id, metric_name="factual_accuracy",
                                    score=int(fa_score * 10), reasoning=str(fa_reason) or None)
                db.add_geval_metric(run_id=self.run_id, metric_name="completeness",
                                    score=int(comp_score * 10), reasoning=str(comp_reason) or None)
                db.add_geval_metric(run_id=self.run_id, metric_name="overall_geval",
                                    score=int(overall_score * 10), reasoning=aggregated_reasoning or None)
            except Exception as e:
                logger.warning(f"Falha ao salvar métricas GEval no banco: {e}")

        return overall_score, aggregated_reasoning

    def _convergence_loop(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        n_runs: int = 10,
        max_retries: int = 3,
        n_runs_min: int = 10,
        min_std_error: float = 0.05,
    ) -> tuple[float, list]:
        retries = successful_runs = n = 0
        scores: list[float] = []
        reasons: list = []

        while retries < max_retries and successful_runs < n_runs:
            try:
                score = metric.measure(test_case)
                scores.append(score)
                reasons.append(metric.reason)
                successful_runs += 1
                n += 1

                std_error = np.std(scores) / np.sqrt(len(scores))
                if n >= n_runs_min and std_error < min_std_error:
                    return float(np.mean(scores)), reasons
            except Exception as e:
                print(f"Erro em {metric.name}: {e}. Tentando novamente...")
                retries += 1

        return (float(np.mean(scores)), reasons) if scores else (0.0, [])

    def _aggregate_reasons(self, reasons: list) -> str:
        prompt = (
            "You are an expert AI assistant specialized in summarizing evaluation feedback.\n"
            "Given multiple reasoning statements from different evaluation runs, aggregate them "
            "into a single coherent reasoning that captures the key points. The aggregated "
            "reasoning must be as general as possible, rather than using specific names or methods.\n\n"
            f"Reasons:\n{reasons}"
        )
        return self.azure_openai.generate(prompt)


class LLMJudgeMetric(Metric):
    """Métrica placeholder que simula um juiz LLM (retorna score aleatório)."""

    def __init__(self, name: str = "llm_judge"):
        super().__init__(name)

    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        scores = [random.uniform(0, 10) for _ in zip(predictions, references)]
        return float(np.mean(np.array(scores)))


# ---------------------------------------------------------------------------
# Geração de datasets
# ---------------------------------------------------------------------------

def parse_users_info_to_specialists(users_info_str: str) -> dict[str, Specialist]:
    """
    Converte a string da coluna 'users_info' do CSV para um dicionário de Specialists.

    Parameters
    ----------
    users_info_str:
        String Python literal representando uma lista de dicts com chaves
        'name', 'type' e opcionalmente 'background_info'.

    Returns
    -------
    dict[str, Specialist]
        Dicionário {nome: Specialist}.
    """
    users_list = ast.literal_eval(users_info_str)
    return {
        u["name"]: Specialist(
            name=u["name"],
            type=u["type"],
            background_info=u.get("background_info"),
        )
        for u in users_list
    }


def generate_pareto_dataset(seed: int = 42) -> list[dict]:
    """
    Gera um dataset de treinamento via simulação EDD (dados de mineração simulados).

    Returns
    -------
    list[dict]
        Lista de dicts com chaves: full_knowledge, run_id, users_with_knowledge,
        question, expected, expected_description.
    """
    table_themes = ["mining"]
    dataset: list[dict] = []

    for theme in table_themes:
        config = EDDRunConfig(
            max_hier_depth=2,
            n_employees=5,
            mean_degree=math.ceil(5 ** 0.5),
            alpha=0.1,
            decay=0.8,
            forgetting_chance=0.7,
            n_patients_zero=1,
            connections=1.5,
            table_info=[(theme, 2, 0.8)],
        )
        run, simulated_users, full_knowledge = edd_simulation(config, seed, theme=theme)

        domain_name = list(full_knowledge.domains.keys())[0]
        domain_description = full_knowledge.domains[domain_name].description
        column_descriptions = full_knowledge.domains[domain_name].facts
        col_text = "\n".join(f"- {k}: {v}" for k, v in column_descriptions.items())

        dataset.append({
            "full_knowledge": full_knowledge,
            "run_id": run.id,
            "users_with_knowledge": simulated_users,
            "question": (
                f"Describe the dataset related to {theme} operations, "
                "including key attributes and their significance."
            ),
            "expected": 10,
            "expected_description": domain_name + "\n" + domain_description + "\n" + col_text,
        })

    return dataset


def generate_hotpot_pareto_dataset(
    n: int = 4,
    seed: int = 42,
    hotpot_path: str | Path | None = None,
) -> list[dict]:
    """
    Gera um dataset de treinamento a partir do HotPotQA (questões de nível 'hard').

    Parameters
    ----------
    n:
        Número de questões a usar (padrão 4).
    seed:
        Semente para reproducibilidade da simulação EDD.
    hotpot_path:
        Caminho para o arquivo hotpot_train_v1.1.json. Se None, usa
        data/hotpot_train_v1.1.json relativo a este módulo.

    Returns
    -------
    list[dict]
        Lista de dicts com chaves: full_knowledge, run_id, users_with_knowledge,
        question, expected, expected_description.
    """
    if hotpot_path is None:
        hotpot_path = _this_dir / "data" / "hotpot_train_v1.1.json"

    with open(hotpot_path) as f:
        data = json.load(f)

    qa = [
        [item["question"], item["context"], item["answer"]]
        for item in data
        if item["level"] == "hard"
    ]

    dataset: list[dict] = []

    for i in range(n):
        theme = f"hotpotqa_question_{i}"
        external_question = qa[i][0]
        expected_answer = qa[i][2]
        raw_data = qa[i][1]

        config = EDDRunConfig(
            max_hier_depth=10,
            n_employees=10,
            mean_degree=math.ceil(5 ** 0.5),
            alpha=0,
            decay=0.8,
            forgetting_chance=0,
            n_patients_zero=1,
            connections=1.5,
            table_info=[("mining", 3, 0.8)],
            pre_existing_data=(raw_data, "HotPotQA", theme),
            external_specialist_role=True,
            single_knowledge_employee=True,
        )
        run, simulated_users, full_knowledge = edd_simulation(config, seed, external_data=True)

        domain_name = list(full_knowledge.domains.keys())[0]
        domain_description = full_knowledge.domains[domain_name].description
        column_descriptions = full_knowledge.domains[domain_name].facts
        col_text = "\n".join(f"- {k}: {v}" for k, v in column_descriptions.items())

        dataset.append({
            "full_knowledge": full_knowledge,
            "run_id": run.id,
            "users_with_knowledge": simulated_users,
            "question": external_question,
            "expected": 10,
            "expected_description": expected_answer,
        })

    return dataset


def get_data_from_df(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """
    Carrega um dataset a partir de um DataFrame com colunas:
    table_name, description, facts, table_theme.

    Returns
    -------
    list[dict]
        Lista de dicts com chaves: full_knowledge, question, expected,
        expected_description.
    """
    dataset: list[dict] = []

    for i, row in enumerate(df.itertuples()):
        if i >= n:
            break
        facts = eval(row.facts)
        full_knowledge = FullKnowledge(title=row.table_name)
        full_knowledge.add_domain(row.table_name, row.description)
        for col, desc in facts.items():
            full_knowledge.add_fact(row.table_name, col, desc)

        domain_description = full_knowledge.domains[row.table_name].description
        column_descriptions = full_knowledge.domains[row.table_name].facts
        col_text = "\n".join(f"- {k}: {v}" for k, v in column_descriptions.items())

        dataset.append({
            "full_knowledge": full_knowledge,
            "question": (
                f"Describe the dataset related to {row.table_theme} operations, "
                "including key attributes and their significance."
            ),
            "expected": 10,
            "expected_description": row.table_theme + "\n" + domain_description + "\n" + col_text,
        })

    return dataset


# ---------------------------------------------------------------------------
# Ponto de entrada — workflow de mineração via CSV
# ---------------------------------------------------------------------------

async def main():
    """
    Workflow principal: carrega dataset do CSV do Kontex e executa a
    otimização GEPA sobre o módulo de questionamento (mining workflow).
    """
    pareto_size = 1
    feedback_size = 1

    csv_path = _parent_dir / "kontex" / "data" / "simulated_table_info.csv"
    dataset: list[dict] = []

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full_knowledge_obj = parse_full_knowledge_from_string(
                row["full_knowledge"],
                FullKnowledge=FullKnowledge,
                DomainKnowledge=DomainKnowledge,
            )
            users_with_knowledge = parse_users_info_to_specialists(row["users_info"])
            dataset.append({
                "full_knowledge": full_knowledge_obj,
                "run_id": UUID(row["run_id"]) if row.get("run_id") else None,
                "users_with_knowledge": users_with_knowledge,
                "question": row.get("question"),
                "expected": int(row.get("expected", 10)),
                "expected_description": row.get("expected_description"),
                "users_info": row.get("users_info"),
            })

    dataset = dataset[: pareto_size + feedback_size]

    system = CompoundAISystem(
        modules={
            "questioning": LanguageModule(
                id="questioning",
                prompt=(
                    "You're helping acquire knowledge about a table by questioning specialists.\n\n"
                    "Current Table Description:\n{table_description}\n\n"
                    "Recent Critique:\n{critique_response}\n\n"
                    "Conversation History with {specialist}:\n{chat_history}\n\n"
                    "Generate a focused question for {specialist} to improve our table understanding.\n"
                    "Focus on:\n"
                    "- Column meanings and data types\n"
                    "- Example values\n"
                    "- Business context and relationships\n\n"
                    "This is very important: DO NOT ASK the specialist for SQL snippets to confirm "
                    "data. You only need the metadata, not the data itself.\n"
                    "Question:"
                ),
                model_weights="gpt-5-mini",
            )
        },
        control_flow=KontexFlow(),
        input_schema=IOSchema(fields={"full_knowledge": FullKnowledge}, required=["full_knowledge"]),
        output_schema=IOSchema(fields={"output": int}, required=["output"]),
        system_id="kontex",
    )

    env = EnvConfig(env_file=str(_this_dir / ".env"))
    api_key = env.api_key
    base_url = env.base_url

    if not api_key:
        print("Defina OPENAI_API_KEY no arquivo .env")
        return

    config = GEPAConfig(
        inference=InferenceConfig(
            provider="openai",
            model="gpt-5-mini",
            api_key=api_key,
            max_tokens=4096,
            temperature=0.1,
            timeout=30,
            base_url=base_url,
            retry_attempts=3,
        ),
        optimization=OptimizationConfig(
            budget=20,
            pareto_set_size=pareto_size,
            minibatch_size=feedback_size,
            enable_crossover=True,
            crossover_probability=0.3,
            mutation_types=["rewrite", "insert"],
        ),
        database=DatabaseConfig(url="sqlite:///gepa_mining.db"),
        observability=ObservabilityConfig(
            log_level="INFO",
            log_file="gepa_mining.log",
            enable_logging=True,
        ),
    )

    evaluator = SimpleFeedbackEvaluator([GEvalMetric(name="geval_metric")])
    inference_client = InferenceFactory.create_client(config.inference)
    optimizer = GEPAOptimizer(config=config, evaluator=evaluator, inference_client=inference_client)

    try:
        result = await optimizer.optimize(system, dataset, max_generations=5)

        print("✅ Otimização concluída!")
        print(f"   Melhor score: {result.best_score:.3f}")
        print(f"   Total rollouts: {result.total_rollouts}")
        print(f"   Custo total: ${result.total_cost:.4f}")

        best_prompt = result.best_system.modules["questioning"].prompt
        print("\n🧠 Prompt de questionamento otimizado:")
        print("-" * 40)
        print(best_prompt)

        out_dir = _this_dir / "optimized_prompts"
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"questioning_optimized_{timestamp}.txt"
        out_file.write_text(best_prompt)
        print(f"\n💾 Prompt salvo em: {out_file}")

    except Exception:
        import traceback
        print(f"❌ Otimização falhou:\n{traceback.format_exc()}")
    finally:
        if hasattr(inference_client, "close"):
            await inference_client.close()
        print("\n🎉 Processo concluído!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
