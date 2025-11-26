#!/usr/bin/env python3
"""
Kontex-GEPA Configuration and Utilities

This file provides configuration classes and utility functions for integrating
GEPA with Kontex for prompt optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import yaml
from pathlib import Path


class OptimizationTarget(Enum):
    """Defines which Kontex agents to optimize."""
    QUESTIONER = "questioner"
    CRITIC = "critic"
    SPECIALIST = "specialist"
    ALL = "all"


class KnowledgeDomain(Enum):
    """Different types of knowledge domains for testing."""
    CUSTOMER_DATA = "customer_data"
    SALES = "sales_transactions"
    INVENTORY = "product_inventory"
    FINANCE = "financial_reports"
    MARKETING = "marketing_campaigns"
    HR = "employee_records"


@dataclass
class KontexOptimizationConfig:
    """Configuration for Kontex-GEPA integration."""
    
    # GEPA Configuration
    gepa_budget: int = 30
    pareto_set_size: int = 8
    minibatch_size: int = 4
    max_generations: int = 6
    
    # Kontex Configuration
    optimization_targets: List[OptimizationTarget] = field(
        default_factory=lambda: [OptimizationTarget.ALL]
    )
    knowledge_domains: List[KnowledgeDomain] = field(
        default_factory=lambda: [
            KnowledgeDomain.CUSTOMER_DATA,
            KnowledgeDomain.SALES,
            KnowledgeDomain.INVENTORY
        ]
    )
    
    # Training Dataset Configuration
    n_training_scenarios: int = 20
    scenario_complexity_levels: List[str] = field(
        default_factory=lambda: ["simple", "medium", "complex"]
    )
    
    # Evaluation Configuration
    knowledge_extraction_weight: float = 0.4
    question_quality_weight: float = 0.3
    critique_accuracy_weight: float = 0.3
    
    # Output Configuration
    save_optimized_prompts: bool = True
    output_directory: str = "./optimized_kontex_prompts"
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'KontexOptimizationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class PromptTemplate:
    """Template for generating domain-specific specialist prompts."""
    
    BASE_SPECIALIST_TEMPLATE = """You are a {domain} specialist with deep knowledge about {domain_description}.

Your expertise includes:
{expertise_areas}

When asked about data tables in your domain, you provide:
- Accurate column meanings and business context
- Data types, formats, and valid value ranges  
- Example values and typical data patterns
- Business rules, constraints, and relationships
- Domain-specific terminology and concepts

Guidelines:
- Be specific and detailed when you know the answer
- Admit when something is outside your expertise
- Suggest who else might know if you can't help
- Use domain-appropriate language and examples
- Focus on practical, actionable information

Domain Context: {domain_context}
"""
    
    ENHANCED_QUESTIONING_TEMPLATE = """You are an expert knowledge acquisition specialist focused on extracting tacit knowledge about data tables.

Your goal is to systematically uncover:
1. **Column semantics**: What each column represents in business terms
2. **Data characteristics**: Types, formats, ranges, and patterns
3. **Business rules**: Constraints, relationships, and dependencies
4. **Contextual knowledge**: How the data is used and interpreted

Questioning Strategy:
- Start with high-level context and purpose
- Drill down into specific columns and their meanings
- Ask for concrete examples and edge cases
- Explore relationships between data elements
- Validate understanding with confirmation questions

Question Types to Use:
- Exploratory: "What does this table represent in your business?"
- Specific: "Can you explain what the 'customer_type' column contains?"
- Example-seeking: "What are some typical values in the 'status' field?"
- Relationship-focused: "How does 'order_date' relate to 'delivery_date'?"
- Validation: "So you're saying this field can only contain X, Y, or Z?"

Remember: The user has mental models and tacit knowledge that needs to be made explicit. Ask questions that help surface implicit assumptions and domain expertise.
"""
    
    ENHANCED_CRITIQUE_TEMPLATE = """You are a rigorous evaluator of table documentation quality and completeness.

Evaluation Criteria (Score 0-10):

**Completeness (40% of score):**
- All columns identified and named
- Data types specified for each column  
- Value ranges and formats documented
- Missing: -2 points per gap

**Semantic Clarity (30% of score):**
- Business meaning of each column explained
- Domain context and purpose clear
- Relationships between columns described
- Vague descriptions: -1 point each

**Practical Usability (20% of score):**
- Example values provided
- Business rules and constraints noted
- Edge cases and exceptions covered
- Data quality expectations described

**Accuracy and Certainty (10% of score):**
- Information appears reliable and specific
- Uncertainty and guesswork clearly flagged
- Sources of information indicated when relevant

Scoring Guidelines:
- 0-2: Minimal information, mostly gaps
- 3-4: Basic structure, major gaps remain
- 5-6: Good foundation, some important details missing
- 7-8: Comprehensive, minor gaps or improvements needed
- 9-10: Excellent, thorough documentation

Be strict but fair. High scores require genuinely complete and useful documentation.
"""
    
    @classmethod
    def generate_specialist_prompt(cls, domain: KnowledgeDomain, 
                                 domain_context: str = "") -> str:
        """Generate a domain-specific specialist prompt."""
        
        domain_configs = {
            KnowledgeDomain.CUSTOMER_DATA: {
                "description": "customer information and demographics",
                "expertise": [
                    "Customer segmentation and classification",
                    "Demographics and behavioral attributes",
                    "Contact information and preferences",
                    "Customer lifecycle stages and status",
                    "Privacy and data protection requirements"
                ]
            },
            KnowledgeDomain.SALES: {
                "description": "sales transactions and revenue data",
                "expertise": [
                    "Transaction processing and order management",
                    "Product codes, pricing, and discounting",
                    "Sales channels and territories",
                    "Payment methods and financial reconciliation",
                    "Sales performance metrics and KPIs"
                ]
            },
            KnowledgeDomain.INVENTORY: {
                "description": "product inventory and supply chain",
                "expertise": [
                    "Product catalogs and SKU management",
                    "Stock levels, reorder points, and forecasting",
                    "Supplier relationships and procurement",
                    "Warehouse locations and logistics",
                    "Quality control and compliance tracking"
                ]
            },
            KnowledgeDomain.FINANCE: {
                "description": "financial reporting and accounting",
                "expertise": [
                    "Chart of accounts and financial classifications",
                    "Revenue recognition and expense categorization",
                    "Budget planning and variance analysis",
                    "Regulatory compliance and audit requirements",
                    "Financial metrics and performance indicators"
                ]
            },
            KnowledgeDomain.MARKETING: {
                "description": "marketing campaigns and customer engagement",
                "expertise": [
                    "Campaign planning and execution",
                    "Channel performance and attribution",
                    "Customer journey mapping and touchpoints",
                    "Content management and personalization",
                    "Marketing metrics and ROI analysis"
                ]
            },
            KnowledgeDomain.HR: {
                "description": "human resources and employee management",
                "expertise": [
                    "Employee records and personal information",
                    "Compensation and benefits administration",
                    "Performance management and reviews",
                    "Training and development tracking",
                    "Compliance and regulatory requirements"
                ]
            }
        }
        
        config = domain_configs.get(domain, {
            "description": "general business data",
            "expertise": ["Data analysis and interpretation"]
        })
        
        expertise_list = "\n".join(f"- {item}" for item in config["expertise"])
        
        return cls.BASE_SPECIALIST_TEMPLATE.format(
            domain=domain.value,
            domain_description=config["description"],
            expertise_areas=expertise_list,
            domain_context=domain_context or f"Working with {config['description']} in a business context."
        )


class ScenarioGenerator:
    """Generates realistic training scenarios for Kontex optimization."""
    
    @staticmethod
    def generate_table_scenarios() -> List[Tuple[str, Dict[str, Any]]]:
        """Generate realistic table scenarios with varying complexity."""
        
        scenarios = [
            # Simple scenarios
            ("customer_basic", {
                "description": "Basic customer information table",
                "columns": ["customer_id", "first_name", "last_name", "email", "phone"],
                "complexity": "simple",
                "domain": KnowledgeDomain.CUSTOMER_DATA,
                "known_columns": 3,
                "total_columns": 5
            }),
            
            # Medium complexity
            ("sales_transactions", {
                "description": "E-commerce sales transaction data",
                "columns": ["transaction_id", "customer_id", "product_id", "quantity", 
                          "unit_price", "total_amount", "transaction_date", "payment_method",
                          "shipping_address", "tax_amount", "discount_applied", "status"],
                "complexity": "medium", 
                "domain": KnowledgeDomain.SALES,
                "known_columns": 8,
                "total_columns": 12
            }),
            
            # Complex scenarios
            ("financial_reporting", {
                "description": "Multi-dimensional financial reporting cube",
                "columns": ["report_id", "fiscal_period", "account_code", "department_id",
                          "cost_center", "gl_amount", "budget_amount", "variance_amount",
                          "currency_code", "exchange_rate", "audit_trail", "approval_status",
                          "consolidation_level", "entity_id", "reporting_standard", 
                          "adjustment_type", "source_system", "load_timestamp"],
                "complexity": "complex",
                "domain": KnowledgeDomain.FINANCE,
                "known_columns": 12,
                "total_columns": 18
            })
        ]
        
        return scenarios
    
    @staticmethod
    def create_partial_knowledge(scenario: Dict[str, Any], 
                               knowledge_percentage: float = 0.6) -> Dict[str, str]:
        """Create partial knowledge about a table scenario."""
        
        columns = scenario["columns"]
        known_count = int(len(columns) * knowledge_percentage)
        known_columns = columns[:known_count]
        
        # Generate realistic column descriptions
        column_descriptions = {}
        for col in known_columns:
            if "id" in col:
                column_descriptions[col] = f"Unique identifier for {col.replace('_id', '').replace('_', ' ')}"
            elif "date" in col or "time" in col:
                column_descriptions[col] = f"Timestamp indicating when {col.replace('_', ' ')} occurred"
            elif "amount" in col or "price" in col:
                column_descriptions[col] = f"Monetary value representing {col.replace('_', ' ')}"
            elif "status" in col:
                column_descriptions[col] = f"Current state or condition of the record"
            else:
                column_descriptions[col] = f"Information about {col.replace('_', ' ')}"
        
        return column_descriptions


class OptimizationResultsHandler:
    """Handles saving and loading optimization results."""
    
    def __init__(self, output_directory: str):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_optimized_prompts(self, optimization_result, config: KontexOptimizationConfig):
        """Save optimized prompts to files."""
        
        results = {
            "metadata": {
                "optimization_score": optimization_result.best_score,
                "total_rollouts": optimization_result.total_rollouts,
                "total_cost": optimization_result.total_cost,
                "pareto_frontier_size": optimization_result.pareto_frontier.size(),
                "config": config.__dict__
            },
            "optimized_prompts": {},
            "original_prompts": {}
        }
        
        # Save each optimized prompt
        for module_id, module in optimization_result.best_system.modules.items():
            results["optimized_prompts"][module_id] = module.prompt
            
            # Save individual prompt files
            prompt_file = self.output_dir / f"{module_id}_optimized.txt"
            with open(prompt_file, 'w') as f:
                f.write(module.prompt)
        
        # Save complete results as JSON
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {self.output_dir}")
        return results_file
    
    def load_optimized_prompts(self) -> Dict[str, str]:
        """Load previously optimized prompts."""
        results_file = self.output_dir / "optimization_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results.get("optimized_prompts", {})
        else:
            return {}


# Example configuration file content
EXAMPLE_CONFIG_YAML = """
# Kontex-GEPA Optimization Configuration

# GEPA Settings
gepa_budget: 40
pareto_set_size: 10
minibatch_size: 5
max_generations: 8

# Optimization Targets
optimization_targets:
  - "all"  # Options: questioner, critic, specialist, all

# Knowledge Domains to Include
knowledge_domains:
  - "customer_data"
  - "sales_transactions" 
  - "product_inventory"
  - "financial_reports"

# Training Dataset
n_training_scenarios: 25
scenario_complexity_levels:
  - "simple"
  - "medium"
  - "complex"

# Evaluation Weights
knowledge_extraction_weight: 0.4
question_quality_weight: 0.3
critique_accuracy_weight: 0.3

# Output Settings
save_optimized_prompts: true
output_directory: "./optimized_kontex_prompts"
"""


if __name__ == "__main__":
    # Example usage
    print("🔧 Kontex-GEPA Configuration Utilities")
    print("=" * 50)
    
    # Create example configuration
    config = KontexOptimizationConfig()
    print(f"📋 Default configuration created")
    print(f"   Budget: {config.gepa_budget}")
    print(f"   Domains: {[d.value for d in config.knowledge_domains]}")
    
    # Generate example specialist prompt
    specialist_prompt = PromptTemplate.generate_specialist_prompt(
        KnowledgeDomain.CUSTOMER_DATA,
        "CRM system with customer demographics and behavior data"
    )
    print(f"\n🤖 Generated specialist prompt preview:")
    print(specialist_prompt[:200] + "...")
    
    # Generate scenarios
    scenarios = ScenarioGenerator.generate_table_scenarios()
    print(f"\n📊 Generated {len(scenarios)} training scenarios")
    
    print("\n✅ Configuration utilities ready for use!")
