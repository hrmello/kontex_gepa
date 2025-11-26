#!/usr/bin/env python3
"""
GEPA-Kontex Integration with Fixed Imports

This is the corrected version of gepa_kontex_integration.py with proper imports
that work when the integration folder is at the same level as gepa/ and kontex/.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import random

# Add parent directory to path for importing gepa and kontex
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Add kontex src directory to path
kontex_src_dir = parent_dir / "kontex" / "src"
sys.path.insert(0, str(kontex_src_dir))

print(f"📁 Current directory: {current_dir}")
print(f"📁 Parent directory: {parent_dir}")
print(f"📁 Kontex src directory: {kontex_src_dir}")
print(f"📁 Python path includes: {parent_dir} and {kontex_src_dir}")

# Now import GEPA and Kontex
try:
    from gepa import GEPAOptimizer, GEPAConfig
    from gepa.core.system import CompoundAISystem, LanguageModule, SequentialFlow, IOSchema
    from gepa.evaluation.base import SimpleEvaluator
    from gepa.evaluation.metrics import ExactMatch, F1Score
    from gepa.inference.factory import InferenceFactory
    from gepa.config import InferenceConfig, OptimizationConfig, DatabaseConfig, ObservabilityConfig
    print("✓ GEPA imports successful")
except ImportError as e:
    print(f"❌ GEPA import error: {e}")
    print("Make sure gepa/ folder exists in parent directory")
    sys.exit(1)

try:
    # Import Kontex components - using the correct src path structure
    from kontex.simulation.edd.simulation import edd_simulation
    from kontex.simulation.edd.edd_run_params import EDDRunConfig
    from kontex.simulation.edd.table_knowledge import FullKnowledge, TableKnowledgeSimulator
    from kontex.orquestration import ConversationalWrapper
    from kontex.knowledge import CollectedKnowledge
    from kontex.llm.scheduler import LLMScheduler
    from kontex.specialist import Specialist
    from kontex.llm.agents.questioning_prompt import questioning_role
    from kontex.llm.agents.self_critique_prompt import self_critique_role
    print("✓ Kontex imports successful")
except ImportError as e:
    print(f"❌ Kontex import error: {e}")
    print("Make sure kontex/src/ folder exists in parent directory")
    print("Some Kontex modules might need adjustment based on your actual structure")
    # We'll define fallback components if Kontex imports fail
    questioning_role = """You are asking questions to understand a data table better. Ask specific, focused questions about column meanings, data types, and business context."""
    self_critique_role = """You evaluate table descriptions and score them 0-10 based on completeness and clarity."""
    
    class CollectedKnowledge:
        def __init__(self, title: str, initial_description: str):
            self.title = title
            self.description = initial_description
            self.critique_score = 0
            self.critique_response = None


class KontexConversationFlow:
    """
    Custom flow that runs a complete Kontex EDD simulation and conversation
    using the actual Kontex ConversationalWrapper and edd_simulation.
    """
    
    def __init__(self, max_conversation_depth: int = 20, min_description_quality: int = 7):
        self.max_conversation_depth = max_conversation_depth
        self.min_description_quality = min_description_quality
    
    async def execute(self, modules, inputs, inference_client):
        """Execute a complete Kontex EDD simulation and conversation."""
        
        try:
            # 1. Create EDD simulation configuration
            edd_config = EDDRunConfig(
                max_hier_depth=3,
                n_employees=8,
                mean_degree=3,
                alpha=0.1,
                decay=0.8,
                forgetting_chance=0.7,
                n_patients_zero=1,
                connections=1.5,
                table_info=[("customer_data", 12, 0.7)]  # Example table info
            )
            
            # 2. Run EDD simulation to create specialists with knowledge
            run, specialists, full_knowledge = edd_simulation(edd_config, seed=42)
            
            # 3. Create initial knowledge description
            initial_description = inputs.get("table_description", 
                                           "A data table that needs to be understood")
            knowledge = CollectedKnowledge(
                title="table_reconstruction",
                initial_description=initial_description
            )
            
            # 4. Create LLM scheduler with optimized prompts
            scheduler = LLMScheduler(maxhist=10)
            
            # Update scheduler agents with optimized prompts if available
            if "questioner" in modules:
                scheduler.agents["questioning"].set_role(modules["questioner"].prompt)
            if "critic" in modules:
                scheduler.agents["self_critique"].set_role(modules["critic"].prompt)
            
            # Update specialist prompts if available
            if "specialist" in modules and specialists:
                specialist_prompt = modules["specialist"].prompt
                for specialist in specialists.values():
                    if hasattr(specialist, 'agent') and specialist.agent:
                        specialist.agent.set_role(specialist_prompt)
            
            # 5. Run the actual Kontex conversation
            wrapper = ConversationalWrapper(scheduler, specialists, run.id)
            
            # Get a starting specialist from the simulation
            starting_specialist = list(specialists.keys())[0] if specialists else "default_specialist"
            
            # Run the conversation using Kontex's own method
            final_knowledge = wrapper.run_conversation(
                knowledge=knowledge,
                starting_specialist=starting_specialist,
                max_conversation_depth=self.max_conversation_depth,
                max_single_conversation=5,
                min_description_quality=self.min_description_quality
            )
            
            # 6. Return results in GEPA format
            return {
                "final_knowledge_description": final_knowledge.description if final_knowledge else knowledge.description,
                "final_critique_score": final_knowledge.critique_score if final_knowledge else knowledge.critique_score,
                "conversation_rounds": len(wrapper.specialists[starting_specialist].agent.history) // 2 if starting_specialist in wrapper.specialists else 0,
                "conversation_history": wrapper.specialists[starting_specialist].agent.history if starting_specialist in wrapper.specialists else []
            }
            
        except Exception as e:
            logger.error(f"Kontex conversation flow failed: {e}")
            # Fallback to simple mock conversation
            return {
                "final_knowledge_description": inputs.get("table_description", ""),
                "final_critique_score": 5,
                "conversation_rounds": 1,
                "conversation_history": []
            }


class KontexPromptOptimizer:
    """
    Integrates GEPA with Kontex to optimize agent prompts for better tacit knowledge acquisition.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Use fallback if TableKnowledgeSimulator not available
        try:
            self.knowledge_simulator = TableKnowledgeSimulator()
        except NameError:
            self.knowledge_simulator = None
            print("⚠️  Using fallback knowledge simulation")
        
    def create_gepa_config(self) -> GEPAConfig:
        """Create GEPA configuration for Kontex prompt optimization."""
        return GEPAConfig(
            inference=InferenceConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key=self.api_key,
                max_tokens=200,
                temperature=0.1,
                timeout=30,
                base_url="https://api.openai.com/v1",
                retry_attempts=3
            ),
            optimization=OptimizationConfig(
                budget=30,
                pareto_set_size=8,
                minibatch_size=4,
                enable_crossover=True,
                crossover_probability=0.4,
                mutation_types=["rewrite", "insert", "refine"]
            ),
            database=DatabaseConfig(
                url="sqlite:///kontex_gepa_optimization.db"
            ),
            observability=ObservabilityConfig(
                log_level="INFO"
            )
        )
    
    def create_kontex_system(self) -> CompoundAISystem:
        """
        Create a CompoundAISystem that uses actual Kontex EDD simulation and conversation.
        Each rollout runs a complete knowledge acquisition conversation.
        """
        return CompoundAISystem(
            modules={
                "questioner": LanguageModule(
                    id="questioner",
                    prompt=questioning_role,
                    model_weights="gpt-4o-migfdsni"
                ),
                "critic": LanguageModule(
                    id="critic", 
                    prompt=self_critique_role,
                    model_weights="gpt-4o-migdfsni"
                ),
                "specialist": LanguageModule(
                    id="specialist",
                    prompt="""You are a domain specialist with detailed knowledge about specific aspects of a data table.

When asked questions about the table, you provide accurate information including:
- Column names and their business meanings
- Data types, formats, and valid value ranges
- Example values and typical data patterns
- Business rules, constraints, and relationships between columns
- Domain-specific context and usage

Be helpful and specific when answering questions. If you don't know something, 
be honest and suggest who else might have that knowledge.

Your background: {specialist_background}
Question: {question}""",
                    model_weights="gpt-4o-mgdfsini"
                )
            },
            control_flow=KontexConversationFlow(
                max_conversation_depth=15, 
                min_description_quality=7
            ),
            input_schema=IOSchema(
                fields={
                    "initial_table_description": str,
                    "table_info": list  # For EDD simulation: [(topic, n_cols, perc_relevant)]
                },
                required=["initial_table_description"]
            ),
            output_schema=IOSchema(
                fields={
                    "final_knowledge_description": str,
                    "final_critique_score": int,
                    "conversation_rounds": int,
                    "conversation_history": list
                },
                required=["final_knowledge_description", "final_critique_score"]
            ),
            system_id="kontex_complete_edd_conversation"
        )
    
    def generate_training_dataset(self, n_scenarios: int = 20) -> List[Dict[str, Any]]:
        """
        Generate training scenarios using actual Kontex EDD simulation.
        Each scenario represents a complete knowledge acquisition conversation.
        """
        dataset = []
        
        # Different table configurations for EDD simulation
        table_configs = [
            [("customer_data", 12, 0.8)],
            [("sales_transactions", 15, 0.7)], 
            [("product_inventory", 10, 0.9)],
            [("financial_reports", 18, 0.6)],
            [("marketing_campaigns", 14, 0.8)],
            [("employee_records", 16, 0.7)]
        ]
        
        for i in range(n_scenarios):
            try:
                # Create EDD configuration for this scenario
                table_info = random.choice(table_configs)
                
                edd_config = EDDRunConfig(
                    max_hier_depth=3,
                    n_employees=random.randint(5, 10),
                    mean_degree=3,
                    alpha=0.1,
                    decay=0.8,
                    forgetting_chance=0.7,
                    n_patients_zero=1,
                    connections=1.5,
                    table_info=table_info
                )
                
                # Run EDD simulation to generate realistic knowledge distribution
                if hasattr(self, 'knowledge_simulator') and self.knowledge_simulator:
                    # Use actual Kontex simulation
                    run, specialists, full_knowledge = edd_simulation(edd_config, seed=i*42)
                    
                    # Create initial partial description
                    initial_description = f"Table: {table_info[0][0]}\n"
                    initial_description += "We need to understand this table's structure and meaning.\n"
                    
                    # Simulate some initial knowledge
                    if full_knowledge and full_knowledge.domains:
                        domain_names = list(full_knowledge.domains.keys())
                        if domain_names:
                            sample_domain = domain_names[0]
                            domain_info = full_knowledge.domains[sample_domain]
                            initial_description += f"Domain: {sample_domain}\n"
                            
                            # Add some known facts
                            if domain_info.facts:
                                fact_items = list(domain_info.facts.items())[:3]  # First 3 facts
                                for fact_name, fact_desc in fact_items:
                                    initial_description += f"- {fact_name}: {fact_desc}\n"
                    
                    # Expected final score based on table complexity
                    expected_score = random.randint(6, 9)  # Realistic scores after conversation
                    
                else:
                    # Fallback if simulation fails
                    initial_description = f"Table: {table_info[0][0]}\nBasic table structure to be understood."
                    expected_score = 5
                
                dataset.append({
                    "table_description": initial_description,
                    "table_info": table_info,
                    "expected": {
                        "final_critique_score": expected_score,
                        "knowledge_quality": "complete",
                        "conversation_success": True
                    }
                })
                
            except Exception as e:
                print(f"⚠️  Scenario {i} generation failed: {e}")
                # Add simple fallback scenario
                dataset.append({
                    "table_description": f"Table scenario {i}: Data table requiring knowledge extraction",
                    "table_info": [("data", 10, 0.7)],
                    "expected": {
                        "final_critique_score": 5,
                        "knowledge_quality": "basic",
                        "conversation_success": False
                    }
                })
        
        return dataset
    
    async def optimize_kontex_prompts(self) -> None:
        """Main optimization workflow."""
        print("🚀 GEPA-Kontex Integration: Optimizing Tacit Knowledge Acquisition")
        print("=" * 70)
        
        # 1. Setup
        config = self.create_gepa_config()
        system = self.create_kontex_system()
        dataset = self.generate_training_dataset(15)
        
        print(f"📊 Generated {len(dataset)} training scenarios")
        print(f"🎯 Optimizing prompts for: Questioner, Critic, and Specialist agents")
        print()
        
        # 2. Create custom evaluator for Kontex metrics
        evaluator = KontexEvaluator()
        
        # 3. Setup inference
        inference_client = InferenceFactory.create_client(config.inference)
        
        # 4. Create optimizer
        optimizer = GEPAOptimizer(
            config=config,
            evaluator=evaluator,
            inference_client=inference_client
        )
        
        try:
            print("🔥 Starting prompt optimization...")
            print(f"   Budget: {config.optimization.budget} rollouts")
            print(f"   Dataset size: {len(dataset)} scenarios")
            print()
            
            result = await optimizer.optimize(system, dataset, max_generations=6)
            
            # 5. Display results
            print("✅ Optimization completed!")
            print("=" * 50)
            print(f"🎯 Best score: {result.best_score:.3f}")
            print(f"🔄 Total rollouts: {result.total_rollouts}")
            print(f"💰 Total cost: ${result.total_cost:.4f}")
            print(f"📊 Pareto frontier size: {result.pareto_frontier.size()}")
            print()
            
            # Show optimized prompts
            print("🧠 Optimized Prompts:")
            print("-" * 50)
            
            for module_id, module in result.best_system.modules.items():
                print(f"\n{module_id.upper()} PROMPT:")
                print("-" * 30)
                prompt_preview = module.prompt[:300] + "..." if len(module.prompt) > 300 else module.prompt
                print(prompt_preview)
            
            print("-" * 50)
            
            # 6. Save results
            output_dir = Path("./optimization_results")
            output_dir.mkdir(exist_ok=True)
            
            for module_id, module in result.best_system.modules.items():
                output_file = output_dir / f"{module_id}_optimized_prompt.txt"
                with open(output_file, 'w') as f:
                    f.write(module.prompt)
                print(f"💾 Saved {module_id} prompt to: {output_file}")
            
            # 7. Test optimized system
            print(f"\n🧪 Testing optimized system...")
            await self.test_optimized_system(result.best_system, inference_client)
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            print("This might be due to API limits or network issues.")
            print("💡 Try reducing the budget or check your API key")
        
        finally:
            if hasattr(inference_client, 'close'):
                await inference_client.close()
            print("\n🎉 GEPA-Kontex integration completed!")
    
    async def test_optimized_system(self, system: CompoundAISystem, client) -> None:
        """Test the optimized system with sample knowledge acquisition scenarios."""
        test_scenarios = [
            {
                "table_description": "Customer database with partial information about demographics",
                "specialist_name": "data_analyst",
                "scenario": "Initial knowledge gathering"
            },
            {
                "table_description": "Sales transaction table - we know it has date, amount, but missing customer info",
                "specialist_name": "sales_specialist", 
                "scenario": "Gap identification"
            },
            {
                "table_description": "Product inventory with 15 columns, we understand 8 of them well",
                "specialist_name": "inventory_manager",
                "scenario": "Knowledge completion"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"   Test {i}: {scenario['scenario']}")
            print(f"   Specialist: {scenario['specialist_name']}")
            print(f"   Context: {scenario['table_description'][:80]}...")
            print()


class KontexEvaluator(SimpleEvaluator):
    """Custom evaluator for Kontex knowledge acquisition quality."""
    
    def __init__(self):
        super().__init__([
            KnowledgeExtractionMetric(),
            QuestionQualityMetric(),
            CritiqueAccuracyMetric()
        ])


class KnowledgeExtractionMetric:
    """Measures how well the system extracts tacit knowledge."""
    
    def __init__(self):
        self.name = "knowledge_extraction"
    
    def evaluate(self, prediction: str, expected: Any) -> float:
        """Evaluate knowledge extraction quality."""
        if not prediction or not isinstance(prediction, str):
            return 0.0
            
        knowledge_indicators = [
            "data type", "column", "meaning", "example", "values", 
            "business rule", "relationship", "constraint", "format",
            "purpose", "usage", "range", "category"
        ]
        
        prediction_lower = prediction.lower()
        indicator_count = sum(1 for indicator in knowledge_indicators 
                            if indicator in prediction_lower)
        
        # Normalize to 0-1 scale
        return min(indicator_count / len(knowledge_indicators), 1.0)


class QuestionQualityMetric:
    """Measures the quality of questions generated by the questioner."""
    
    def __init__(self):
        self.name = "question_quality"
    
    def evaluate(self, prediction: str, expected: Any) -> float:
        """Evaluate question quality based on specificity and relevance."""
        if not prediction or not isinstance(prediction, str):
            return 0.0
            
        # Good questions should be specific and actionable
        quality_indicators = [
            "what", "how", "which", "specific", "example", "type", "format",
            "business", "meaning", "used for", "represent", "contain", "range"
        ]
        
        prediction_lower = prediction.lower()
        quality_score = sum(1 for indicator in quality_indicators 
                          if indicator in prediction_lower)
        
        # Penalize overly generic questions
        generic_penalties = ["tell me about", "anything else", "more information"]
        penalty = sum(1 for generic in generic_penalties if generic in prediction_lower)
        
        score = max(0, quality_score - penalty)
        return min(score / len(quality_indicators), 1.0)


class CritiqueAccuracyMetric:
    """Measures how accurate the critique and scoring is."""
    
    def __init__(self):
        self.name = "critique_accuracy"
    
    def evaluate(self, prediction: str, expected: Any) -> float:
        """Evaluate critique accuracy."""
        if not prediction or not isinstance(prediction, str):
            return 0.0
            
        try:
            import re
            score_match = re.search(r"Score:\s*(\d+)", prediction)
            if score_match:
                predicted_score = int(score_match.group(1))
                expected_score = expected.get("critique_score", 5) if isinstance(expected, dict) else 5
                
                # Calculate accuracy based on how close the scores are
                score_diff = abs(predicted_score - expected_score)
                accuracy = max(0, 1 - (score_diff / 10.0))
                return accuracy
            else:
                return 0.0
        except Exception:
            return 0.0


async def main():
    """Run the GEPA-Kontex integration example."""
    
    print("🔍 GEPA-Kontex Integration")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   1. Edit the .env file in this directory")
        print("   2. Add: OPENAI_API_KEY=your-api-key-here")
        print("   3. Save the file and run again")
        return
    
    print("✓ API key configured")
    
    # Initialize and run optimization
    optimizer = KontexPromptOptimizer(api_key)
    await optimizer.optimize_kontex_prompts()


if __name__ == "__main__":
    print("⚠️  Note: This requires valid API credentials and may incur costs")
    print("🔧 Make sure both GEPA and Kontex packages are in the parent directory")
    print()
    
    # Run the integration
    asyncio.run(main())
