from pathlib import Path
from dotenv import load_dotenv
import random
import sys
import math
from uuid import UUID
from sentence_transformers import SentenceTransformer, util

# Add parent directory to path for importing gepa and kontex
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Add kontex src directory to path
kontex_src_dir = parent_dir / "kontex" / "src"
sys.path.insert(0, str(kontex_src_dir))    

from kontex.logging import logger
from kontex.simulation.edd.simulation import edd_simulation
from kontex.simulation.edd.edd_run_params import EDDRunConfig
from kontex.simulation.edd.table_knowledge import FullKnowledge, TableKnowledgeSimulator
from kontex.orquestration import ConversationalWrapper
from kontex.knowledge import CollectedKnowledge
from kontex.llm.scheduler import LLMScheduler
from kontex.specialist import Specialist
from kontex.llm.agents.questioning_prompt import questioning_role
from kontex.llm.agents.self_critique_prompt import self_critique_role

from gepa import GEPAOptimizer, GEPAConfig
from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
from gepa.evaluation.base import SimpleEvaluator
from gepa.evaluation.metrics import ExactMatch, F1Score
from gepa.inference.factory import InferenceFactory
from gepa.config import InferenceConfig, OptimizationConfig, DatabaseConfig, ObservabilityConfig


class KontexControlFlow:

    def __init__(self):
        pass    

    def run_conversation_simulation(self,
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

            # scheduler.agents["questioning"].
            conversational_wrapper = ConversationalWrapper(
                scheduler,
                simulated_users,
                run_id,
            )
            initial_user = rng.choice(list(simulated_users.keys()))
            description = conversational_wrapper.run_conversation(
                table,
                initial_user,
                min_description_quality=9,
            )

            logger.info(f"Final Table Description:\n{description}")
            logger.info(
                f"\n-------------\nOriginal Description: \n{table_knowledge.facts}"
            )

            descriptions[table_name] = description
        return descriptions


    def execute(self):
        seed = 42
        rng = random.Random(seed)
        current_data = dict()

        config = EDDRunConfig(
            max_hier_depth=2,
            n_employees=5,
            mean_degree=math.ceil(5 ** (1 / 2)),
            alpha=0.1,
            decay=0.8,
            forgetting_chance=0.7,
            n_patients_zero=1,
            connections=1.5,
            table_info=[("mining", 3, 0.8)],
        )

        run, simulated_users, full_knowledge = edd_simulation(config, seed)

        print(full_knowledge)
        print(f"Run completed with {len(simulated_users)} users.")
       
        description = self.run_conversation_simulation(
                            simulated_users=simulated_users,
                            full_knowledge=full_knowledge,
                            seed=seed,
                            run_id=run.id,
                            )

        current_data["final_table_description"] = description

        return current_data
    
questioner_prompt = """
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
        """
        
        # Default critique prompt  
critique_prompt =  """
        Evaluate the completeness of this table description:
        
        {table_description}
        
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

    
def run_conversation_simulation(
        run_id: UUID,
        simulated_users: dict[str, Specialist],
        full_knowledge: FullKnowledge,
        initial_prompts: dict[str, str],
        seed: int = None
    ) -> dict[str, str]:
        rng = random.Random(seed)

        # TODO verificar como iremos lidar com múltiplas tabelas no futuro (se o agente tenta encontrar tudo de uma vez ou explora uma tabela por vez)
        descriptions = {}
        for table_name, table_knowledge in full_knowledge.domains.items():
            table_columns = list(table_knowledge.facts.keys())
            initial_description = f"Table: {table_name}\nColumns: {table_columns}"
            table = CollectedKnowledge(table_name, initial_description)

            scheduler = LLMScheduler(maxhist=0)  # Only use the most recent messages

            # scheduler.agents["questioning"].
            conversational_wrapper = ConversationalWrapper(
                initial_prompts = initial_prompts,
                scheduler = scheduler,
                simulated_users = simulated_users,
                run_id = run_id,
            )
            initial_user = rng.choice(list(simulated_users.keys()))
            description = conversational_wrapper.run_conversation(
                table,
                initial_user,
                min_description_quality=9,
            )

            logger.info(f"Final Table Description:\n{description}")
            logger.info(
                f"\n-------------\nOriginal Description: \n{table_knowledge.facts}"
            )

            descriptions[table_name] = description
        return descriptions


def generate_pareto_dataset(seed = 42):
    from collections import defaultdict
    table_themes = ["mining", "healthcare", "finance", "technology", "retail", "education"]
    
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

        print("RUN", run)
    
        domain_name = list(full_knowledge.domains.keys())[0]
        print("DOMAIN NAME", domain_name)
        domain_description = full_knowledge.domains[domain_name].description
        column_descriptions = full_knowledge.domains[domain_name].facts

        theme_dict = dict()
        theme_dict["full_knowledge"] = full_knowledge
        theme_dict["theme"] = theme
        theme_dict["domain_description"] = domain_description
        theme_dict["column_descriptions"] = column_descriptions
        theme_dict["users_with_knowledge"] = simulated_users
        theme_dict["question"] = f"Describe the dataset related to {theme} operations, including key attributes and their significance."
        theme_dict["final_score"] = 10

        dataset.append(theme_dict)

    return dataset
    # metric to be used to evolve GEPA will be numeric, by comparing the expected final score with the one the critic uses
    # if the difference between the two decreased from previous gepa iteration, then the prompt is better
    # TODO: need to include the similarity metric in the final_score calculation, becaususe the critic only 
    # evaluates how the answer is written and not so much its content

    # TODO: Use reasoning of deepeval metrics to create final_score 

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

if __name__ == "__main__":

    budget = 0
    total_budget = 10
    dataset, full_knowledge = generate_pareto_dataset()

    print(dataset)
    dpareto_size = 4
    dpareto = dataset[:dpareto_size]
    dfeedback = dataset[dpareto_size:]

    prompts = {
        "questioner_prompt": questioner_prompt,
        "critique_prompt": critique_prompt
    }

    while budget < total_budget:

        descriptions_dataset, scores_dataset = evaluate_prompt_kontex(prompts, dpareto)

        add_to_pareto_frontier(scores_dataset, prompts)
        budget += 1

    # kontex = KontexControlFlow()
    # description = kontex.execute()
    # print(description)