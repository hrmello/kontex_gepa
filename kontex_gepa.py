from typing import Any, Dict, List
from pathlib import Path
from dotenv import load_dotenv
import random
import sys, os
import math
import numpy as np
from uuid import UUID
from sentence_transformers import SentenceTransformer, util
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import BaseMetric, GEval
from deepeval.metrics.g_eval import Rubric

# Add parent directory to path for importing gepa and kontex
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

print(parent_dir)
# Add kontex src directory to path
kontex_src_dir = parent_dir / "kontex" / "src"
gepa_dir = parent_dir / "gepa" / "src" 

sys.path.insert(0, str(kontex_src_dir))    
sys.path.insert(0, str(gepa_dir))  

print(sys.path)
from kontex.logging import logger
# from kontex.database import db
from kontex.knowledge import CollectedKnowledge
from kontex.simulation.edd.table_knowledge import FullKnowledge
from kontex.llm.scheduler import LLMScheduler
from kontex.specialist import Specialist
from kontex.simulation.edd.simulation import edd_simulation
from kontex.simulation.edd.edd_run_params import EDDRunConfig
from kontex.orquestration import ConversationalWrapper

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, IOSchema
from gepa.evaluation.base import SimpleEvaluator, SimpleFeedbackEvaluator
from gepa.inference.factory import InferenceFactory
from gepa.config import InferenceConfig, OptimizationConfig, DatabaseConfig, ObservabilityConfig
from gepa.evaluation.metrics import Metric

def run_conversation_simulation(
    prompts: dict[str, str],
    run_id: UUID,
    simulated_users: dict[str, Specialist],
    full_knowledge: FullKnowledge,
    seed: int = None,
    min_description_quality: int = 9,
) -> dict[str, str]:
    rng = random.Random(seed)

    # TODO verificar como iremos lidar com múltiplas tabelas no futuro (se o agente tenta encontrar tudo de uma vez ou explora uma tabela por vez)
    descriptions = {}
    for table_name, table_knowledge in full_knowledge.domains.items():
        table_columns = list(table_knowledge.facts.keys())
        initial_description = f"Table: {table_name}\nColumns: {table_columns}"
        table = CollectedKnowledge(table_name, initial_description)

        scheduler = LLMScheduler(maxhist=0)  # Only use the most recent messages
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
            min_description_quality=min_description_quality,
            max_conversation_depth = 15 # Limit the conversation depth to avoid long runtimes during testing
        )

        logger.info(f"Final Table Description:\n{description}")
        logger.info(
            f"\n-------------\nOriginal Description: \n{table_knowledge.facts}"
        )
        logger.info(f"Final Critique Score: {final_critique_score}")
        descriptions[table_name] = description
    return descriptions, final_critique_score

class EnvConfig:
    """Configuration class to manage environment variables"""
    
    def __init__(self, env_file=".env"):
        # Load environment variables from .env file
        env_path = Path(env_file)
        
        if env_path.exists():
            load_dotenv(env_path)
            # print(load_dotenv(env_path))
            print(f"✓ Loaded environment variables from {env_file}")
        else:
            print(f"⚠ Warning: {env_file} file not found")
        
        # Load all configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_MODEL")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT")

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"



class KontexFlow:
    """A placeholder for KontexFlow control flow logic."""
    
    def __init__(self, min_description_quality: int = 8):
        self.min_description_quality = min_description_quality
        
    async def execute(
        self,
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any
    ) -> Dict[str, Any]:
        """Execute modules in a predefined KontexFlow manner."""

        current_data = input_data.copy()

        logger.info("Prompts: \n\n -Questioning Module Prompt:\n")
        logger.info(modules["questioning"].prompt)
        # logger.info("\n -Critique Module Prompt:\n")
        # logger.info(modules["critique"].prompt)
        prompts = {
            "questioning_prompt": modules["questioning"].prompt,
            # "critique_prompt": modules["critique"].prompt
            "critique_prompt": """
                Evaluate the completeness of this table description:
                        
                         {tacit_knowledge}
                        
                         Provide assessment in this format:
                         Score: [0-10]
                         Reasoning: [why this score]
                         Suggestions: [what's missing]
                        
                         To score high (8+), description needs:
                         - All column names and meanings
                         - Data types for each column
                         - Example values where relevant
                         - Business context and purpose

                """
            }

        current_data = input_data.copy()

        description, final_critique_score = run_conversation_simulation(
            run_id=input_data.get("run_id", UUID(int=0)),
            prompts=prompts,
            simulated_users=input_data["users_with_knowledge"],
            full_knowledge=input_data["full_knowledge"],
            seed=42,
            min_description_quality=self.min_description_quality,
        )

        # Ensure we return a numeric value
        if final_critique_score is None:
            final_critique_score = 0.0
        elif not isinstance(final_critique_score, (int, float)):
            try:
                final_critique_score = float(final_critique_score)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert final_critique_score to float: {final_critique_score}")
                final_critique_score = 0.0

        # current_data['description'] = description
        current_data['output'] = final_critique_score
        current_data['predicted_description'] = description
        logger.debug(f"Final critic score: {current_data['output']}")
        return current_data
    
class AverageDiffScore(Metric):
    """Average Difference Score Metric."""
    
    def __init__(self, name: str = "score"):
        super().__init__(name)
       
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute exact match score."""
        
        logger.debug(f"Predictions: {predictions}")
        logger.debug(f"References: {references}")

        scores = []
        for pred, ref in zip(predictions, references):
            diff = 10 - (ref - pred["output"])  # max score is 10
            scores.append(diff)

        logger.debug(f"Scores: {scores}")
        logger.debug(f"Mean score: {np.mean(np.array(scores))}")
        return np.mean(np.array(scores))

class GEvalMetric(Metric):
    """
    Metric that makes use of different criteria
    """

    def __init__(self,  name: str = "geval_metric"):
        super().__init__(name)
        self.name = name

        config = EnvConfig(env_file = ".env")
        # Check for API key
        self.api_key = config.api_key
        self.model = config.model
        azure_endpoint = config.azure_endpoint
        openai_api_version = "2025-01-01-preview"
        azure_deployment = "gpt-5-mini"

    # Replace these with real values
        custom_model = AzureChatOpenAI(
            model = self.model,
            azure_endpoint = azure_endpoint,
            azure_deployment=azure_deployment,
            openai_api_key = self.api_key,
            openai_api_version = openai_api_version,
        )

        self.azure_openai = AzureOpenAI(model=custom_model)

    def compute(self, predictions: list, references: list):
        """
        Compute several criteria scores between prediction and reference descriptions.
        """
        weight_hallucination = 0.6
        weight_completeness = 0.4

        for pred, ref in zip(predictions, references):
            # print("Ref", ref)
            # print("Predictions:", pred.keys())
            prediction_description = pred["predicted_description"]
            assert len(list(prediction_description.keys())) == 1, "Currently only single table descriptions are supported."
            prediction_description = prediction_description[list(prediction_description.keys())[0]]

            expected_description = ref["expected_description"]

            # print("Prediction description:", prediction_description)
            # print("Expected description:", expected_description)
            prediction_score = pred["output"]
            expected_score = ref["expected"]

            factual_accuracy = GEval(
                name="Factual Accuracy",
                model = self.azure_openai,
                criteria="Evaluate whether the actual output contains any made-up, incorrect, or fabricated facts when compared to the expected output. Penalize heavily for invented information.",
                #TODO Testar evaluation_steps
                # avalie se tem alucinação. Caso tenha, não dê score maior que X
                rubric = [
                    Rubric(score_range=(0,4), expected_outcome="Mostly made-up or incorrect content."),
                    Rubric(score_range=(5,7), expected_outcome=f"Half correct, a considerable amount of made-up content. Around 30-50% of the content is fabricated."),
                    Rubric(score_range=(8,9), expected_outcome="Mostly correct with few fabricated content (less than 20%)."),
                    Rubric(score_range=(10,10), expected_outcome=f"100% correct."),
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                threshold=0.7
            )

            #TODO Usar métricas "fixas" da literatura: BERTScore, BLEU, ROUGE, etc
            completeness = GEval(
                name="Completeness",
                model = self.azure_openai,
                criteria="Evaluate how much of the key information from the expected output is covered in the actual output. Check for missing variables, descriptions, or important details.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                threshold=0.7
            )
        
            # print("Prediction description:", prediction_description[list(prediction_description.keys())[0]])
            test_case = LLMTestCase(
                input="Provide a comprehensive description of a table, including detailed variable descriptions with their data types, purposes, expected ranges, common issues, validation rules, and relationships to other tables.",
                actual_output=prediction_description,
                expected_output=expected_description,
                retrieval_context=[expected_description] 
            )

            
            factual_accuracy_scores, factual_accuracy_reason = self.convergence_geval_loop(factual_accuracy, test_case, n_runs = 20, max_retries = 3, min_std_error=0.05, n_runs_min=5)
            completeness_scores, completeness_reason = self.convergence_geval_loop(completeness, test_case, n_runs = 20, max_retries = 3, min_std_error=0.05, n_runs_min=5)

            # TODO Usar uma LLM aberta pra pegar as probabilidades (subir no nebius/GCP)
            computed_statistics_factual = self.compute_statistics(factual_accuracy_scores)
            computed_statistics_completeness = self.compute_statistics(completeness_scores)

            overall_score = (weight_hallucination*computed_statistics_factual["mean"] + weight_completeness*computed_statistics_completeness["mean"])

            aggregated_reasoning = self.aggregate_reasons([factual_accuracy_reason, completeness_reason])

            return overall_score, aggregated_reasoning
    
    def aggregate_reasons(self, reasons: List[str]) -> str:

        prompt_aggregation = f"""
        You are an expert AI assistant specialized in summarizing evaluation feedback.
        Given multiple reasoning statements from different evaluation runs, your task is to aggregate them into a single coherent reasoning that captures the key points.  The aggregated reasoning must be as general as possible, rather than using specific names or methods.
        
        Here is a list of reasons for different evaluation metrics:
        {reasons}"""

        reasoning_aggregation = self.azure_openai.generate(prompt_aggregation)

        return reasoning_aggregation

    
    def compute_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Compute mean and standard deviation of scores."""
        mean_score = np.mean(np.array(scores))
        std_deviation = np.std(np.array(scores))

        return {
            "mean": mean_score,
            "std_deviation": std_deviation,
            "std_error": std_deviation / np.sqrt(len(scores))
        }
    
    def convergence_geval_loop(self, metric: BaseMetric, test_case: LLMTestCase, n_runs: int = 10, max_retries: int = 3, n_runs_min: int = 10, min_std_error: float = 0.05):

        retries = 0
        sucessful_runs = 0
        scores = list()
        reasoning = list()
        n = 0
        
        while retries < max_retries and sucessful_runs < n_runs:
            try:
                print(f"{metric.name} run {n}")
                score = metric.measure(test_case)
                scores.append(score)
                print(f"{metric.name}:", score)
                sucessful_runs += 1
                n += 1

                partial_std_deviation = np.std(scores)
                partial_std_error = partial_std_deviation / np.sqrt(len(scores))
                
                reasoning.append(metric.reason)
                print(f"Standard Error in run {n}: {partial_std_error}")
                
                if n >= n_runs_min and partial_std_error < min_std_error: # Confidence Interval = 0.95 if min_std_error = 0.05
                    print(f"{metric.name} converged after {n} runs.")
                    return scores, reasoning
                
            except Exception as e:
                print(f"Error during {metric.name} evaluation: {e}. Retrying...")
                retries += 1
                continue

        print(f"{metric.name} did not converge. Returning {len(scores)} scores.")
        return scores, reasoning

class LLMJudgeMetric(Metric):
    """LLM Judge Metric."""
    
    def __init__(self, name: str = "llm_judge"):
        super().__init__(name)
       
    
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute exact match score."""
        
        logger.debug(f"Predictions: {predictions}")
        logger.debug(f"References: {references}")

        scores = []
        for pred, ref in zip(predictions, references):
            # Here we would call an LLM to judge the quality of pred against ref
            # For simplicity, we'll use a dummy score
            judge_score = random.uniform(0, 10)  # Dummy score between 0 and 10
            scores.append(judge_score)

        logger.debug(f"Scores: {scores}")
        logger.debug(f"Mean score: {np.mean(np.array(scores))}")
        return np.mean(np.array(scores))
    

def evaluate_prompt_kontex(prompts:dict, dataset: dict):

    descriptions_dataset = list()
    scores_dataset = list()
    for datapoint in dataset:
        description = run_conversation_simulation(
            initial_prompts=prompts,
            simulated_users=datapoint["users_with_knowledge"],
            full_knowledge=datapoint["full_knowledge"],
            seed=42,
        )
    
        descriptions_dataset.append(description)

        similarity_matrix, score = compute_similarity(description, datapoint)

        scores_dataset.append(score)
    return descriptions_dataset, scores_dataset

def compute_similarity(description, datapoint):
    """
    Compute the semantic similarity between the description and the facts in full_knowledge.
    """

    desc_texts = list(description.values())
    domain = list(datapoint["full_knowledge"].domains.keys())[0]
    facts_texts = list(datapoint["full_knowledge"].domains[domain].facts.values())

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings
    desc_emb = model.encode(desc_texts, convert_to_tensor=True)
    facts_emb = model.encode(facts_texts, convert_to_tensor=True)

    # Cosine simlarity
    similarity_matrix = util.cos_sim(desc_emb, facts_emb)

    # Aggregate the scores (e.g., mean of max similarities for each description)
    score = similarity_matrix.max(dim=1).values.mean().item()

    return similarity_matrix, score



def generate_pareto_dataset(seed = 42, numbers_of_themes: int = 3, max_hier_depth: int = 2, n_employees: int = 5, n_columns_per_table: int = 3) -> List[Dict[str, Any]]:
    from collections import defaultdict
    import random 
    def join_description_and_facts(description: str, facts: Dict[str, str]) -> str:
        combined = description.strip() + "\n\nFacts:\n"
        for key, val in facts.items():
            combined += f"- {key}: {val.strip()}\n\n"
        return combined

    all_table_themes = ["mining", "healthcare", "finance", "technology", "retail", "education"]
    
    table_themes = random.Random(seed).sample(all_table_themes, min(numbers_of_themes, len(all_table_themes)))

    dataset = list()

    for theme in table_themes:
        config = EDDRunConfig(
                max_hier_depth=max_hier_depth,
                n_employees=n_employees,
                mean_degree=math.ceil(5 ** (1 / 2)),
                alpha=0.1,
                decay=0.8,
                forgetting_chance=0.7,
                n_patients_zero=1,
                connections=1.5,
                table_info=[(theme, n_columns_per_table, 0.8)], #(theme, 3, 0.8)
            )

        run, simulated_users, full_knowledge = edd_simulation(config, seed)
    
        domain_name = list(full_knowledge.domains.keys())[0]
        domain_description = full_knowledge.domains[domain_name].description
        column_descriptions = full_knowledge.domains[domain_name].facts

        theme_dict = dict()
        theme_dict["full_knowledge"] = full_knowledge
        theme_dict["run_id"] = run.id
        theme_dict["users_with_knowledge"] = simulated_users
        theme_dict["question"] = f"Describe the dataset related to {theme} operations, including key attributes and their significance."
        theme_dict["expected"] = 10
        theme_dict["expected_description"] = join_description_and_facts(domain_description, column_descriptions)

        logger.debug(f"Table description: {full_knowledge.domains[domain_name].description}")
        logger.debug(f"Column descriptions: {full_knowledge.domains[domain_name].facts}")

        dataset.append(theme_dict)

    return dataset
    # metric to be used to evolve GEPA will be numeric, by comparing the expected final score with the one the critic uses
    # if the difference between the two decreased from previous gepa iteration, then the prompt is better
    # TODO: need to include the similarity metric in the final_score calculation, becaususe the critic only 
    # evaluates how the answer is written and not so much its content

    # TODO: Use reasoning of deepeval metrics to create final_score 

async def main():

    # dataset parameters
    NUMBER_OF_THEMES = 3
    MAX_HIER_DEPTH = 3
    N_EMPLOYEES = 8
    N_COLUMNS_PER_TABLE = 2

    # kontex conversation parameters
    MIN_DESCRIPTION_QUALITY = 8

    # gepa parameters
    PARETO_SET_SIZE = 2
    MINI_BATCH_SIZE = NUMBER_OF_THEMES - PARETO_SET_SIZE

    
    logger.info(f"Parameters used in this run:")
    logger.info(f"  NUMBER_OF_THEMES: {NUMBER_OF_THEMES}")
    logger.info(f"  MAX_HIER_DEPTH: {MAX_HIER_DEPTH}")
    logger.info(f"  N_EMPLOYEES: {N_EMPLOYEES}")
    logger.info(f"  N_COLUMNS_PER_TABLE: {N_COLUMNS_PER_TABLE}")
    logger.info(f"  MIN_DESCRIPTION_QUALITY: {MIN_DESCRIPTION_QUALITY}")
    logger.info(f"  PARETO_SET_SIZE: {PARETO_SET_SIZE}")
    logger.info(f"  MINI_BATCH_SIZE: {MINI_BATCH_SIZE}")
    
    # 1. Creating Kontex dataset
    dataset = generate_pareto_dataset(numbers_of_themes=NUMBER_OF_THEMES, 
                                      max_hier_depth=MAX_HIER_DEPTH, 
                                      n_employees=N_EMPLOYEES, 
                                      n_columns_per_table=N_COLUMNS_PER_TABLE)

    # 2. System with 2 modules: questioner and critique
    system = CompoundAISystem(
        modules={
            "questioning": LanguageModule(
                id="questioning",
                prompt="""
                        You're helping acquire knowledge about a table by questioning specialists.
                        
                        Current Table Description:
                        {table_description}
                        
                        Recent Critique:
                        {critique_response}
                        
                        Conversation History with {specialist}:
                        {chat_history}
                        
                        Generate a focused question for {specialist} to improve our table understanding.
                        Focus on:
                        - Column meanings and data types
                        - Example values
                        - Business context and relationships
                        
                        Do not ask the {specialist} to create SQL queries for any reason. Make explicit to {specialist} not to write any SQL queries.
                         
                        Question:
                        """,
                model_weights="gpt-5-mini"
            )
            ## TODO: Comentei a mudança no prompt to critico pra ver como se comporta somente com o questionador evoluindo
            # ,

            # "critique": LanguageModule(
            #     id="critique",
            #     prompt="""
            #             Evaluate the completeness of this table description:
                        
            #             {tacit_knowledge}
                        
            #             Provide assessment in this format:
            #             Score: [0-10]
            #             Reasoning: [why this score]
            #             Suggestions: [what's missing]
                        
            #             To score high (8+), description needs:
            #             - All column names and meanings
            #             - Data types for each column
            #             - Example values where relevant
            #             - Business context and purpose
            #             """,
            #     model_weights="gpt-5-mini"
            # ),
        },
        control_flow=KontexFlow(min_description_quality=MIN_DESCRIPTION_QUALITY),
        input_schema=IOSchema(
            fields={"full_knowledge": FullKnowledge},
            required=["full_knowledge"]
        ),
        output_schema=IOSchema(
            fields={"output": int},
            required=["output"]
        ),
        system_id="kontex"
    )

    config = EnvConfig(env_file = ".env")
    # Check for API key
    api_key = config.api_key
    base_url = config.base_url
    # print("api key:", api_key)
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("Found API key")
    # 3. Configuration
    config = GEPAConfig(
        inference=InferenceConfig(
            provider="openai",
            model="gpt-5-mini",
            api_key=api_key,
            max_tokens=4096,
            temperature=0.1,
            timeout=30,
            base_url=base_url,
            retry_attempts=3
        ),
        optimization=OptimizationConfig(
            budget=20,
            pareto_set_size=PARETO_SET_SIZE, #change pareto set size 
            minibatch_size=MINI_BATCH_SIZE,
            enable_crossover=True,
            crossover_probability=0.3,
            mutation_types=["rewrite", "insert"]
        ),
        database=DatabaseConfig(
            url="sqlite:///gepa_quickstart.db"
        ),
        observability=ObservabilityConfig(
            log_level="INFO",
            log_file="gepa_quickstart.log",
            enable_logging=True
        )
    )

    # 4. Create evaluator (need to change metrics for Kontex)
    evaluator = SimpleFeedbackEvaluator([
        # AverageDiffScore(name="average_score")
        GEvalMetric(name="geval_metric")
    ])
    
    # 5. Create inference client
    print(config.inference.provider)
    inference_client = InferenceFactory.create_client(config.inference)

    # 6. Create optimizer and run optimization
    print("🔄 Starting optimization...")
    print(f"   Budget: {config.optimization.budget} rollouts")
    print(f"   Dataset size: {len(dataset)} examples")
    print()
    
    optimizer = GEPAOptimizer(
        config=config,
        evaluator=evaluator,
        inference_client=inference_client
    )
    
    try:
        result = await optimizer.optimize(system, dataset, max_generations=5)
        
        # 7. Display results
        print("✅ Optimization completed!")
        print("=" * 50)
        print(f"🎯 Best score: {result.best_score:.3f}")
        print(f"🔄 Total rollouts: {result.total_rollouts}")
        print(f"💰 Total cost: ${result.total_cost:.4f}")
        print(f"📊 Pareto frontier size: {result.pareto_frontier.size()}")
        print()
        
        # Show the optimized prompt
        best_questioning_module = result.best_system.modules["questioning"]
        # best_critique_module = result.best_system.modules["critique"]
        print("🧠 Optimized questioning prompt:")
        print("-" * 30)
        print(best_questioning_module.prompt)

        logger.info(f"Best questioning prompt: \n {best_questioning_module.prompt}")
        # logger.info(f"Best critique prompt: \n {best_critique_module.prompt}")
        print("🧠 Optimized critiqu prompt:")
        print("-" * 30)
        # print(best_critique_module.prompt)
        print("-" * 30)
        print()
        
        # # Test the optimized system
        # print("🧪 Testing optimized system...")
        # test_examples = [
        #     "This movie was absolutely incredible!",
        #     "I'm disappointed with this purchase.",
        #     "The weather is fine today."
        # ]
        
        # for test_text in test_examples:
        #     try:
        #         # Simulate running the optimized system
        #         input_data = {"text": test_text}
        #         # In a real scenario, you'd run: result = await result.best_system.execute(input_data, inference_client)
        #         # For demo, we'll just show the input
        #         print(f"   Input: '{test_text}'")
        #         print(f"   System: sentiment_classifier")
        #         print()
        #     except Exception as e:
        #         print(f"   Error testing: {e}")
        
        # Show optimization statistics
        stats = optimizer.get_statistics()
        print("📊 Optimization Statistics:")
        print(f"   Generations completed: {stats.get('generations', 0)}")
        print(f"   Successful mutations: {stats.get('successful_mutations', 0)}")
        print(f"   Average score improvement: {stats.get('average_improvement', 0):.3f}")
        
    except Exception as e:
        import traceback
        print(f"❌ Optimization failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("This might be due to API limits or network issues.")
        print("Try again with a smaller budget or check your API key.")
    
    finally:
        # Clean up
        await inference_client.close() if hasattr(inference_client, 'close') else None
        print("\n🎉 Quickstart example completed!")

    # prompts = {
    #     "questioner_prompt": questioner_prompt,
    #     "critique_prompt": critique_prompt
    # }

    # descriptions_dataset, scores_dataset = evaluate_prompt_kontex(prompts, dpareto)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    #TODO: Implementar métrica que avalie a velocidade de convergencia do GEPA ao longo das gerações. Se o prompt diminuir o número de interações para uma boa nota, melhor.