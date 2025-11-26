from pathlib import Path
import random
from typing import Any, Dict, List
import sys
from uuid import UUID

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

from gepa.core.system import LanguageModule

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
            min_description_quality=6,
            max_conversation_depth = 20 # Limit the conversation depth to avoid long runtimes during testing
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

def generate_pareto_dataset(seed = 42):
    from collections import defaultdict
    table_themes = ["mining"]
    
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
        logger.info("\n -Critique Module Prompt:\n")
        logger.info(modules["critique"].prompt)
        prompts = {
            "questioning_prompt": modules["questioning"].prompt,
            "critique_prompt": modules["critique"].prompt
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
    
if __name__ == "__main__":
    flow = KontexFlow()
