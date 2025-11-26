from pathlib import Path
from dotenv import load_dotenv
import random
import sys
import math
from uuid import UUID


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

async def main():
    system = CompoundAISystem(
            modules = {
                "questioner": LanguageModule(
                    id="questioner",
                    prompt=questioner_prompt,
                    input_schema=IOSchema(
                        fields={
                            "table_description": str,
                            "specialist": str,
                            "chat_history": str,
                            "critique_response": str
                        },
                        required=["table_description", "specialist"]
                    )
                ),
                "critique": LanguageModule(
                    id="critique",
                    prompt=critique_prompt,
                    input_schema=IOSchema(
                        fields={"table_description": str},
                        required=["table_description"]
                    )
                )
            },
            control_flow = KontexControlFlow(),
            input_schema = IOSchema(
                    fields={
                        "initial_description": str,
                        "max_interactions": int
                    },
                    required=[]
                    ),
            output_schema = IOSchema(
                fields={
                    "final_table_description": dict[str, str],
                    # "final_critique_score": int,
                    # "total_interactions": int,
                    # "specialists_contacted": list,
                    # "chat_histories": dict,
                    # "critique_history": list
                },
                required=["final_table_description"]
            ),
            system_id="kontex"
        )
    
if __name__ == "__main__":

    kontex = KontexControlFlow()
    description = kontex.execute()
    print(description)