"""
GEPA-Kontex Integration Package

This package integrates GEPA (Genetic Pareto Prompt Optimizer) with Kontex 
for optimizing tacit knowledge acquisition prompts.
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path to access gepa and kontex packages
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Add kontex src directory to path (kontex uses src layout)
kontex_src_dir = parent_dir / "kontex" / "src"
sys.path.insert(0, str(kontex_src_dir))

# Verify that GEPA and Kontex are available
try:
    import gepa
    print("✓ GEPA package found")
except ImportError as e:
    print(f"❌ GEPA package not found: {e}")
    print(f"   Make sure GEPA folder exists at: {parent_dir / 'gepa'}")

try:
    import kontex
    print("✓ Kontex package found")
except ImportError as e:
    print(f"❌ Kontex package not found: {e}")
    print(f"   Make sure Kontex folder exists at: {parent_dir / 'kontex'}")

# Version info
__version__ = "0.1.0"
__author__ = "GEPA-Kontex Integration"

# Export main classes for easy import
try:
    from .gepa_kontex_integration import KontexPromptOptimizer
    from .kontex_gepa_config import KontexOptimizationConfig, PromptTemplate, KnowledgeDomain
    
    __all__ = [
        "KontexPromptOptimizer",
        "KontexOptimizationConfig", 
        "PromptTemplate",
        "KnowledgeDomain"
    ]
except ImportError as e:
    print(f"⚠️  Some integration modules not available: {e}")
    __all__ = []