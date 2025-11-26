#!/usr/bin/env python3
"""
Kontex-GEPA Quickstart

A simplified example showing how to optimize Kontex agent prompts using GEPA.
This example follows the same pattern as the original GEPA quickstart.py.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory and parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))  # Current directory first
sys.path.insert(0, str(parent_dir))   # Parent directory for gepa/kontex

print(f"📁 Running from: {current_dir}")
print(f"📁 Parent directory: {parent_dir}")

# Import the integration components we created
try:
    from kontex_optimization import KontexPromptOptimizer
    print("✓ KontexPromptOptimizer imported")
except ImportError as e:
    print(f"❌ Failed to import KontexPromptOptimizer: {e}")
    print("Make sure gepa_kontex_integration.py is in the current directory")
    sys.exit(1)

try:
    from kontex_gepa_config import KontexOptimizationConfig, PromptTemplate, KnowledgeDomain
    print("✓ Configuration classes imported")
except ImportError as e:
    print(f"❌ Failed to import config classes: {e}")
    print("Make sure kontex_gepa_config.py is in the current directory")
    sys.exit(1)


async def simple_kontex_optimization():
    """Run a simple Kontex prompt optimization using GEPA."""
    
    print("\n🚀 Kontex-GEPA Quickstart")
    print("=" * 50)
    
    # 1. Check API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   1. Edit the .env file in this directory")
        print("   2. Replace 'your-openai-api-key-here' with your actual API key")
        print("   3. Save and run again")
        return
    
    print("✓ API key found")
    
    # 2. Create simple configuration
    print("\n⚙️  Setting up optimization...")
    
    config = KontexOptimizationConfig(
        gepa_budget=15,  # Reduced for quickstart
        pareto_set_size=5,
        minibatch_size=3,
        max_generations=3,
        n_training_scenarios=10,  # Fewer scenarios for speed
        knowledge_domains=[KnowledgeDomain.CUSTOMER_DATA, KnowledgeDomain.SALES],
        output_directory="./quickstart_results"
    )
    
    print(f"   Budget: {config.gepa_budget} rollouts")
    print(f"   Scenarios: {config.n_training_scenarios}")
    print(f"   Domains: {[d.value for d in config.knowledge_domains]}")
    
    # 3. Initialize optimizer
    optimizer = KontexPromptOptimizer(api_key)
    
    # 4. Show original prompts
    print("\n📄 Original Kontex Prompts (preview):")
    print("-" * 40)
    
    system = optimizer.create_kontex_system()
    questioner_preview = system.modules["questioner"].prompt[:150]
    print(f"Questioner: {questioner_preview}...")
    
    specialist_preview = system.modules["specialist"].prompt[:150]  
    print(f"Specialist: {specialist_preview}...")
    
    # 5. Run optimization
    print(f"\n🔥 Starting optimization...")
    print("   This may take a few minutes...")
    
    try:
        await optimizer.optimize_kontex_prompts()
        
        print("\n✅ Quickstart completed successfully!")
        print("🔍 Check ./quickstart_results/ for optimized prompts")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("💡 Try reducing the budget or check your API key")
    
    print("\n🎉 Quickstart finished!")


async def demonstrate_prompt_generation():
    """Demonstrate the prompt generation capabilities."""
    
    print("\n🎭 Prompt Generation Demo")
    print("=" * 40)
    
    # Show different specialist prompts for different domains
    domains = [KnowledgeDomain.CUSTOMER_DATA, KnowledgeDomain.SALES, KnowledgeDomain.FINANCE]
    
    for domain in domains:
        print(f"\n{domain.value.upper()} Specialist:")
        print("-" * 25)
        
        prompt = PromptTemplate.generate_specialist_prompt(domain)
        # Show first few lines
        lines = prompt.split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"  {line}")
        print("  ...")


async def show_training_data_example():
    """Show an example of the training data generation."""
    
    print("\n📊 Training Data Example")
    print("=" * 30)
    
    # Create a simple optimizer to generate example data
    optimizer = KontexPromptOptimizer("dummy_key")  # Just for data generation
    dataset = optimizer.generate_training_dataset(1)  # Just 3 examples
    
    print(dataset)
    for i, example in enumerate(dataset, 1):
        print(f"\nExample {i}:")
        print(f"  Table: {example['table_description'][:80]}...")
        print(f"  Specialist: {example['specialist_name']}")
        print(f"  Expected score: {example['expected']['critique_score']}")


if __name__ == "__main__":
    print("🔬 Kontex-GEPA Integration Quickstart")
    print("This demonstrates how to optimize Kontex agent prompts using GEPA")
    print()
    
    # Show what we can do
    print("Available demos:")
    print("1. Full optimization (requires API key)")
    print("2. Prompt generation demonstration")
    print("3. Training data example")
    print()
    
    # For safety, we'll run the demos that don't require API calls
    print("Running demos 2 and 3 (no API calls)...")
    
   
    asyncio.run(demonstrate_prompt_generation())
    asyncio.run(show_training_data_example())
    
    print("\n" + "="*50)
    print("To run the full optimization, uncomment the line below:")
    print("# asyncio.run(simple_kontex_optimization())")
    print()
    print("⚠️  Note: Full optimization requires OPENAI_API_KEY and will incur costs")
    print("📖 Check the configuration options in kontex_gepa_config.py")
    
    # Uncomment this line to run full optimization:
    asyncio.run(simple_kontex_optimization())
    

    # print(f"❌ Demo failed: {e}")
    # print("Please check that all files are in the correct directory")
