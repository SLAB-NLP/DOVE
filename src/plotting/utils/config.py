"""
Configuration file for LLM evaluation plotting scripts.
Contains all shared config and configurations for different scripts.
"""

import os
from pathlib import Path

# =============================================================================
# USER CONFIGURATION SECTION - Modify these values as needed
# =============================================================================

# HuggingFace Authentication
# Set your HuggingFace token here OR set HF_ACCESS_TOKEN environment variable
# DO NOT commit your actual token to version control! Use environment variables instead.
HF_ACCESS_TOKEN = "None"  # Will be loaded from environment variables
HF_ACCESS_TOKEN="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"
# Cache Directory Configuration
# Directory where downloaded data will be cached (set to None to disable caching)
DEFAULT_CACHE_DIR = "/Users/ehabba/PycharmProjects/DOVE/Data/dove_lite_data"
CACHE_DIR_ENV_VAR = "DOVE_CACHE_DIR"

# Output Directory Configuration
# Base directory where all plots will be saved
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] /  "plots"

# Specific output directories for different plot types
OUTPUT_DIRS = {
    'accuracy_marginalization': DEFAULT_OUTPUT_DIR / "accuracy_marginalization",
    'few_shot_variance': DEFAULT_OUTPUT_DIR / "few_shot_variance", 
    'performance_variations': DEFAULT_OUTPUT_DIR / "performance_variations",
    'success_rate_distribution': DEFAULT_OUTPUT_DIR / "success_rate_distribution",
}

def get_cache_directory() -> str:
    """
    Get the cache directory from environment variable or use default from config.
    
    Returns:
        Path to the cache directory
    """
    return os.getenv(CACHE_DIR_ENV_VAR, DEFAULT_CACHE_DIR)

def get_output_directory(plot_type: str = None) -> Path:
    """
    Get the output directory for a specific plot type or the base directory.
    
    Args:
        plot_type: Type of plot ('accuracy_marginalization', 'few_shot_variance', etc.)
                  If None, returns the base output directory
    
    Returns:
        Path object for the output directory
    """
    if plot_type is None:
        return DEFAULT_OUTPUT_DIR
    return OUTPUT_DIRS.get(plot_type, DEFAULT_OUTPUT_DIR / plot_type)

# Model Configurations - Add or remove models as needed
DEFAULT_MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'allenai/OLMoE-1B-7B-0924-Instruct',
    'meta-llama/Llama-3.3-70B-Instruct',
]

# Dataset Configurations - Choose which datasets to analyze
# Quick test setup (recommended for first time users - fast and lightweight):
DEFAULT_DATASETS = [
    'social_iqa',
    'mmlu.college_biology',
]

# For comprehensive analysis with all 77 datasets:
# 1. Comment out the quick setup above 
# 2. Uncomment the full list below:
# WARNING: Full analysis takes much longer and requires more memory!
"""
DEFAULT_DATASETS = [
    # Full dataset list (comprehensive analysis):
    # Base datasets
    "ai2_arc.arc_challenge",
    "ai2_arc.arc_easy", 
    "hellaswag",
    "openbook_qa",
    "social_iqa",
    'quality',
    # MMLU subtasks
    "mmlu.abstract_algebra", "mmlu.anatomy", "mmlu.astronomy", "mmlu.business_ethics",
    "mmlu.clinical_knowledge", "mmlu.college_biology", "mmlu.college_chemistry",
    "mmlu.college_computer_science", "mmlu.college_mathematics", "mmlu.college_medicine",
    "mmlu.college_physics", "mmlu.computer_security", "mmlu.conceptual_physics",
    "mmlu.econometrics", "mmlu.electrical_engineering", "mmlu.elementary_mathematics",
    "mmlu.formal_logic", "mmlu.global_facts", "mmlu.high_school_biology",
    "mmlu.high_school_chemistry", "mmlu.high_school_computer_science",
    "mmlu.high_school_european_history", "mmlu.high_school_geography",
    "mmlu.high_school_government_and_politics", "mmlu.high_school_macroeconomics",
    "mmlu.high_school_mathematics", "mmlu.high_school_microeconomics",
    "mmlu.high_school_physics", "mmlu.high_school_psychology",
    "mmlu.high_school_statistics", "mmlu.high_school_us_history",
    "mmlu.high_school_world_history", "mmlu.human_aging", "mmlu.human_sexuality",
    "mmlu.international_law", "mmlu.jurisprudence", "mmlu.logical_fallacies",
    "mmlu.machine_learning", "mmlu.management", "mmlu.marketing", "mmlu.medical_genetics",
    "mmlu.miscellaneous", "mmlu.moral_disputes", "mmlu.moral_scenarios", "mmlu.nutrition",
    "mmlu.philosophy", "mmlu.prehistory", "mmlu.professional_accounting",
    "mmlu.professional_law", "mmlu.professional_medicine", "mmlu.professional_psychology",
    "mmlu.public_relations", "mmlu.security_studies", "mmlu.sociology",
    "mmlu.us_foreign_policy", "mmlu.virology", "mmlu.world_religions",
    # MMLU Pro subtasks
    "mmlu_pro.history", "mmlu_pro.law", "mmlu_pro.health", "mmlu_pro.physics", 
    "mmlu_pro.business", "mmlu_pro.other", "mmlu_pro.philosophy", "mmlu_pro.psychology", 
    "mmlu_pro.economics", "mmlu_pro.math", "mmlu_pro.biology", "mmlu_pro.chemistry", 
    "mmlu_pro.computer_science", "mmlu_pro.engineering",
]
"""

# Processing Parameters - Adjust based on your system capabilities
DEFAULT_SHOTS = [0, 5]  # Few-shot settings to test
DEFAULT_NUM_PROCESSES = 2  # Number of parallel processes (adjust based on your CPU/RAM)
DEFAULT_USE_CACHE = True  # Enable caching for faster subsequent runs

# =============================================================================
# SYSTEM CONFIGURATION SECTION - Usually no need to modify these
# =============================================================================

# All available datasets (comprehensive list - used internally)
ALL_DATASETS = DEFAULT_DATASETS

# Model display names for plots (shorter with line breaks)
MODEL_DISPLAY_NAMES = {
    'meta-llama/Llama-3.2-1B-Instruct': 'Llama-3.2-1B-\nInstruct',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3.2-3B-\nInstruct',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama-3-8B-\nInstruct',
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral-7B-\nInstruct-v0.3',
    'allenai/OLMoE-1B-7B-0924-Instruct': 'OLMoE-1B-7B\n0924-Instruct', 
    'meta-llama/Llama-3.3-70B-Instruct': 'Llama-3.3-70B-\nInstruct',
}

# Color scheme for models (consistent across all plots)
MODEL_COLOR_SCHEME = {
    'meta-llama/Llama-3.2-1B-Instruct': "#1f77b4",  # Blue
    'meta-llama/Llama-3.2-3B-Instruct': "#d62728",   # Red
    'meta-llama/Meta-Llama-3-8B-Instruct': "#2ca02c",  # Green
    'mistralai/Mistral-7B-Instruct-v0.3': "#9467bd",  # Purple
    'allenai/OLMoE-1B-7B-0924-Instruct': "#ff7f0e",  # Orange
    'meta-llama/Llama-3.3-70B-Instruct': "#17becf",  # Cyan
}

# Plot style configurations
PLOT_STYLE = {
    'font_family': 'serif',
    'font_serif': ['DejaVu Serif'],
    'mathtext_fontset': 'dejavuserif',
    'figure_dpi': 600,
    'save_dpi': 300,
    'bbox_inches': 'tight',
    'transparent': False,
    'facecolor': 'white',
}

# =============================================================================
# UTILITY FUNCTIONS - Internal use
# =============================================================================

def get_model_display_name(model_name: str) -> str:
    """Get the display name for a model"""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.split('/')[-1])

def get_model_color(model_name: str) -> str:
    """Get the color for a model"""
    return MODEL_COLOR_SCHEME.get(model_name, "#333333")

def format_dataset_name(dataset_string: str) -> str:
    """
    Format dataset names for display in plots (single line format).
    Unified function to replace all dataset formatting across the codebase.
    
    Args:
        dataset_string: Original dataset name
        
    Returns:
        Formatted dataset name for display (single line)
    """
    # Special case mappings (single line as requested)
    special_mappings = {
        # Base datasets
        'ai2_arc.arc_challenge': "ARC Challenge",
        'ai2_arc.arc_easy': "ARC Easy", 
        'hellaswag': "HellaSwag",
        'openbook_qa': "OpenBook QA",
        'social_iqa': "Social IQA",
        'quality': "QuALITY",
        
        # MMLU datasets
        'mmlu.abstract_algebra': "MMLU Abstract Algebra",
        'mmlu.anatomy': "MMLU Anatomy",
        'mmlu.astronomy': "MMLU Astronomy",
        'mmlu.business_ethics': "MMLU Business Ethics",
        'mmlu.clinical_knowledge': "MMLU Clinical Knowledge",
        'mmlu.college_biology': "MMLU College Biology",
        'mmlu.college_chemistry': "MMLU College Chemistry",
        'mmlu.college_computer_science': "MMLU College Computer Science",
        'mmlu.college_mathematics': "MMLU College Mathematics",
        'mmlu.college_medicine': "MMLU College Medicine",
        'mmlu.college_physics': "MMLU College Physics",
        'mmlu.computer_security': "MMLU Computer Security",
        'mmlu.conceptual_physics': "MMLU Conceptual Physics",
        'mmlu.econometrics': "MMLU Econometrics",
        'mmlu.electrical_engineering': "MMLU Electrical Engineering",
        'mmlu.elementary_mathematics': "MMLU Elementary Mathematics",
        'mmlu.formal_logic': "MMLU Formal Logic",
        'mmlu.global_facts': "MMLU Global Facts",
        'mmlu.high_school_biology': "MMLU High School Biology",
        'mmlu.high_school_chemistry': "MMLU High School Chemistry",
        'mmlu.high_school_computer_science': "MMLU High School Computer Science",
        'mmlu.high_school_european_history': "MMLU High School European History",
        'mmlu.high_school_geography': "MMLU High School Geography",
        'mmlu.high_school_government_and_politics': "MMLU High School Government & Politics",
        'mmlu.high_school_macroeconomics': "MMLU High School Macroeconomics",
        'mmlu.high_school_mathematics': "MMLU High School Mathematics",
        'mmlu.high_school_microeconomics': "MMLU High School Microeconomics",
        'mmlu.high_school_physics': "MMLU High School Physics",
        'mmlu.high_school_psychology': "MMLU High School Psychology",
        'mmlu.high_school_statistics': "MMLU High School Statistics",
        'mmlu.high_school_us_history': "MMLU High School US History",
        'mmlu.high_school_world_history': "MMLU High School World History",
        'mmlu.human_aging': "MMLU Human Aging",
        'mmlu.human_sexuality': "MMLU Human Sexuality",
        'mmlu.international_law': "MMLU International Law",
        'mmlu.jurisprudence': "MMLU Jurisprudence",
        'mmlu.logical_fallacies': "MMLU Logical Fallacies",
        'mmlu.machine_learning': "MMLU Machine Learning",
        'mmlu.management': "MMLU Management",
        'mmlu.marketing': "MMLU Marketing",
        'mmlu.medical_genetics': "MMLU Medical Genetics",
        'mmlu.miscellaneous': "MMLU Miscellaneous",
        'mmlu.moral_disputes': "MMLU Moral Disputes",
        'mmlu.moral_scenarios': "MMLU Moral Scenarios",
        'mmlu.nutrition': "MMLU Nutrition",
        'mmlu.philosophy': "MMLU Philosophy",
        'mmlu.prehistory': "MMLU Prehistory",
        'mmlu.professional_accounting': "MMLU Professional Accounting",
        'mmlu.professional_law': "MMLU Professional Law",
        'mmlu.professional_medicine': "MMLU Professional Medicine",
        'mmlu.professional_psychology': "MMLU Professional Psychology",
        'mmlu.public_relations': "MMLU Public Relations",
        'mmlu.security_studies': "MMLU Security Studies",
        'mmlu.sociology': "MMLU Sociology",
        'mmlu.us_foreign_policy': "MMLU US Foreign Policy",
        'mmlu.virology': "MMLU Virology",
        'mmlu.world_religions': "MMLU World Religions",
        
        # MMLU Pro datasets
        'mmlu_pro.history': "MMLU-Pro History",
        'mmlu_pro.law': "MMLU-Pro Law",
        'mmlu_pro.health': "MMLU-Pro Health",
        'mmlu_pro.physics': "MMLU-Pro Physics",
        'mmlu_pro.business': "MMLU-Pro Business",
        'mmlu_pro.other': "MMLU-Pro Other",
        'mmlu_pro.philosophy': "MMLU-Pro Philosophy",
        'mmlu_pro.psychology': "MMLU-Pro Psychology",
        'mmlu_pro.economics': "MMLU-Pro Economics",
        'mmlu_pro.math': "MMLU-Pro Math",
        'mmlu_pro.biology': "MMLU-Pro Biology",
        'mmlu_pro.chemistry': "MMLU-Pro Chemistry",
        'mmlu_pro.computer_science': "MMLU-Pro Computer Science",
        'mmlu_pro.engineering': "MMLU-Pro Engineering",
    }
    
    # Return mapped name if exists, otherwise format automatically
    if dataset_string in special_mappings:
        return special_mappings[dataset_string]
    
    # Auto-format for unmapped datasets
    if '.' in dataset_string:
        prefix, suffix = dataset_string.split('.', 1)
        suffix_formatted = suffix.replace('_', ' ').title()
        prefix_formatted = prefix.upper() if prefix.lower() in ['mmlu', 'arc'] else prefix.title()
        return f"{prefix_formatted} {suffix_formatted}"
    
    # Simple case: just title case with underscores replaced
    return dataset_string.replace('_', ' ').title() 