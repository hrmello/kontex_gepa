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
from kontex.database import db
from kontex.knowledge import CollectedKnowledge
from kontex.simulation.edd.table_knowledge import FullKnowledge
from kontex.llm.scheduler import LLMScheduler
from kontex.llm.agents import DummyAgent
from kontex.settings import settings
from kontex.specialist import Specialist
from kontex.simulation.edd.simulation import edd_simulation
from kontex.simulation.edd.edd_run_params import EDDRunConfig
from kontex.orquestration import ConversationalWrapper
from kontex.settings import settings

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.inference.factory import InferenceFactory
from gepa.config import InferenceConfig, OptimizationConfig, DatabaseConfig, ObservabilityConfig
from gepa.evaluation.base import Evaluator, EvaluationResult, SimpleEvaluator
from gepa.evaluation.metrics import Metric

def run_conversation_simulation(
    prompts: dict[str, str],
    run_id: UUID,
    simulated_users: dict[str, Specialist],
    full_knowledge: FullKnowledge,
    seed: int = None,
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
            min_description_quality=9,
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
        )
        logger.debug("Final critique score: {final_critique_score}")

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
        azure_endpoint = "https://azureopenai4k.openai.azure.com/"
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
    def compute(self, prediction_description: str, reference_description: str) -> float:
        """
        Compute several criteria scores between prediction and reference descriptions.
        """
        weight_hallucination = 0.6
        weight_completeness = 0.4

        factual_accuracy = GEval(
            name="Factual Accuracy",
            model = self.azure_openai,
            criteria="Evaluate whether the actual output contains any made-up, incorrect, or fabricated facts when compared to the expected output. Penalize heavily for invented information.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.7
        )

        completeness = GEval(
            name="Completeness",
            model = self.azure_openai,
            criteria="Evaluate how much of the key information from the expected output is covered in the actual output. Check for missing variables, descriptions, or important details.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.7
        )
    
        test_case = LLMTestCase(
            input="Provide a comprehensive description of the MineProcessAssays table, including detailed variable descriptions for GRDFe_A, RCV_PCT, and SMP_RUNID with their data types, purposes, expected ranges, common issues, validation rules, and relationships to other tables.",
            actual_output=prediction_description,
            expected_output=reference_description,
            retrieval_context=[reference_description] 
        )

        factual_accuracy_score = factual_accuracy.measure(test_case)
        completeness_score = completeness.measure(test_case)
        overall_score = (weight_hallucination*factual_accuracy_score + weight_completeness*completeness_score)

        return overall_score

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

def generate_pareto_dataset(seed = 42):
    from collections import defaultdict
    table_themes = ["mining", "healthcare"] #"finance", "technology", "retail"]#, "education"]
    
    dataset = list()

    for theme in table_themes:
        config = EDDRunConfig(
                max_hier_depth=2,
                n_employees=5,
                mean_degree=math.ceil(5 ** (1 / 2)),
                alpha=0.1,
                decay=0.8,
                forgetting_chance=0.7,
                n_patients_zero=1,
                connections=1.5,
                table_info=[(theme, 3, 0.8)],
            )

        run, simulated_users, full_knowledge = edd_simulation(config, seed)
    
        domain_name = list(full_knowledge.domains.keys())[0]
        print("DOMAIN NAME", domain_name)
        domain_description = full_knowledge.domains[domain_name].description
        column_descriptions = full_knowledge.domains[domain_name].facts

        theme_dict = dict()
        theme_dict["full_knowledge"] = full_knowledge
        theme_dict["run_id"] = run.id
        theme_dict["users_with_knowledge"] = simulated_users
        theme_dict["question"] = f"Describe the dataset related to {theme} operations, including key attributes and their significance."
        theme_dict["expected"] = 10

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

    # 1. Creating Kontex dataset
    dataset = generate_pareto_dataset()

    print(dataset)
    # dpareto_size = 4
    # dpareto = dataset[:dpareto_size]
    # dfeedback = dataset[dpareto_size:]

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
        control_flow=KontexFlow(),
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
            pareto_set_size=1, #change pareto set size 
            minibatch_size=1,
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
    evaluator = SimpleEvaluator([
        AverageDiffScore(name="average_score")
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