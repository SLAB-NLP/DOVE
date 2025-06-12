"""
Unified plotting utilities to eliminate code duplication across plotting scripts.

This module contains common functionality shared across:
- performance_analysis.py
- few_shot_comparison.py  
- robustness_analysis.py
- prompt_impact_analysis.py
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt

from src.plotting.utils.auth import ensure_hf_authentication
from src.plotting.utils.config import (
    DEFAULT_MODELS, DEFAULT_DATASETS, DEFAULT_SHOTS, DEFAULT_NUM_PROCESSES,
    PLOT_STYLE
)


def setup_matplotlib():
    """Initialize matplotlib with non-interactive backend."""
    matplotlib.use('Agg')  # Non-interactive backend - no display windows


def setup_plot_style():
    """Apply consistent plot styling from config."""
    plt.rcParams['font.family'] = PLOT_STYLE['font_family']
    plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
    plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

    # Some scripts need additional style parameters
    if 'mathtext_fontset' in PLOT_STYLE:
        plt.rcParams['mathtext.fontset'] = PLOT_STYLE['mathtext_fontset']


def setup_figure_style(fig, ax):
    """Apply consistent figure and axis styling."""
    # White background instead of transparent
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.set_facecolor('white')

    # Remove plot borders for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def create_safe_filename(name: str) -> str:
    """Convert dataset/model names to safe filenames."""
    return name.replace('.', '_').replace('/', '_')


def create_output_directory(output_dir: str, subdirectory: str = None) -> str:
    """Create output directory structure and return the path."""
    if subdirectory:
        safe_subdir = create_safe_filename(subdirectory)
        full_dir = f'{output_dir}/{safe_subdir}'
    else:
        full_dir = output_dir

    Path(full_dir).mkdir(parents=True, exist_ok=True)
    return full_dir


def save_figure(output_path: str, formats: List[str] = ['png', 'svg']):
    """
    Save figure in multiple formats with consistent styling.
    
    Args:
        output_path: Base path without extension
        formats: List of formats to save (default: ['png', 'svg'])
    """
    save_kwargs = {
        'bbox_inches': PLOT_STYLE['bbox_inches'],
        'transparent': PLOT_STYLE['transparent'],
        'facecolor': PLOT_STYLE['facecolor']
    }

    for fmt in formats:
        if fmt == 'png':
            plt.savefig(f'{output_path}.png',
                        dpi=PLOT_STYLE.get('save_dpi', PLOT_STYLE['figure_dpi']),
                        **save_kwargs)
        elif fmt == 'pdf':
            plt.savefig(f'{output_path}.pdf', **save_kwargs)
        elif fmt == 'svg':
            plt.savefig(f'{output_path}.svg', **save_kwargs)


def check_existing_files(output_paths: List[str], force_overwrite: bool = False) -> bool:
    """
    Check if output files already exist.
    
    Args:
        output_paths: List of file paths to check
        force_overwrite: Whether to force overwrite existing files
        
    Returns:
        True if should skip (files exist and not forcing), False if should proceed
    """
    if force_overwrite:
        return False

    return all(os.path.exists(path) for path in output_paths)


def filter_models_with_data(data, models: List[str], dataset_name: str = None,
                            required_shots: List[int] = None) -> List[str]:
    """
    Filter models that have data for the specified criteria.
    
    Args:
        data: DataFrame containing evaluation results
        models: List of model names to check
        dataset_name: Optional dataset name to filter by
        required_shots: Optional list of required shot counts
        
    Returns:
        List of models that have data matching the criteria
    """
    models_with_data = []

    for model in models:
        model_data = data[data['model'] == model]

        if model_data.empty:
            if dataset_name:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")
            else:
                print(f"⚠️  No data for model {model} - skipping this model")
            continue

        # Check shots if specified
        if required_shots:
            available_shots = set(model_data['dimensions_5: shots'].unique())
            required_shots_set = set(required_shots)

            if not available_shots.intersection(required_shots_set):
                if dataset_name:
                    print(
                        f"⚠️  Model {model} has no data for required shots {required_shots} for {dataset_name} - skipping")
                else:
                    print(f"⚠️  Model {model} has no data for required shots {required_shots} - skipping")
                continue

            # Print warning if some shots are missing but still have partial data
            missing_shots = required_shots_set - available_shots
            if missing_shots and dataset_name:
                print(
                    f"ℹ️  Model {model} missing shots {missing_shots} for {dataset_name} - will show available shots only")

        models_with_data.append(model)

    return models_with_data


def create_lighter_color(hex_color: str, factor: float = 0.35) -> str:
    """
    Create a lighter version of the given hex color.
    
    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        factor: Lightening factor (0.0 = no change, 1.0 = white)
        
    Returns:
        Lighter hex color string
    """
    rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    lighter_rgb = tuple(int(c + (255 - c) * factor) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)


def create_darker_color(hex_color: str, factor: float = 0.7) -> str:
    """
    Create a darker version of the given hex color.
    
    Args:
        hex_color: Hex color string (e.g., '#1f77b4')
        factor: Darkening factor (0.0 = black, 1.0 = no change)
        
    Returns:
        Darker hex color string
    """
    rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    darker_rgb = tuple(int(c * factor) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*darker_rgb)


def create_base_argument_parser(description: str, epilog: str = None) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.
    
    Args:
        description: Description for the parser
        epilog: Optional epilog text with examples
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )

    # Common arguments across all plotting scripts
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'List of datasets to analyze (default: {len(DEFAULT_DATASETS)} datasets)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help=f'Number of parallel processes (default: {DEFAULT_NUM_PROCESSES})')

    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (default: cache enabled)')

    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing plot files (default: skip existing)')

    parser.add_argument('--list-models', action='store_true',
                        help='List available default models and exit')

    parser.add_argument('--list-datasets', action='store_true',
                        help='List available default datasets and exit')

    return parser


def handle_list_arguments(args) -> bool:
    """
    Handle --list-models and --list-datasets arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if a list option was handled (should exit), False otherwise
    """
    if args.list_models:
        print("Default models:")
        for i, model in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {model}")
        return True

    if args.list_datasets:
        print("Default datasets:")
        for i, dataset in enumerate(DEFAULT_DATASETS, 1):
            print(f"  {i}. {dataset}")
        return True

    return False


def print_analysis_header(analysis_name: str, models: List[str], datasets: List[str],
                          shots: List[int], num_processes: int, use_cache: bool,
                          force_overwrite: bool, extra_info: str = None):
    """
    Print standardized analysis header with configuration details.
    
    Args:
        analysis_name: Name of the analysis being run
        models: List of models
        datasets: List of datasets  
        shots: List of shot counts
        num_processes: Number of processes
        use_cache: Whether cache is enabled
        force_overwrite: Whether force overwrite is enabled
        extra_info: Optional extra information to display
    """
    print(f"Starting {analysis_name}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Shots: {shots}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    if extra_info:
        print(extra_info)
    print("=" * 60)


def print_analysis_footer(analysis_name: str, total_plots: int, output_dir: str):
    """
    Print standardized analysis completion footer.
    
    Args:
        analysis_name: Name of the analysis that completed
        total_plots: Total number of plots created
        output_dir: Output directory where plots were saved
    """
    print("=" * 60)
    print(f"✓ {analysis_name} completed!")
    print(f"  - Created {total_plots} plots")
    print(f"  - Plots saved to: {output_dir}")
    print("=" * 60)


def setup_analysis_environment():
    """
    Set up the analysis environment with authentication and matplotlib.
    Should be called at the start of each main function.
    """
    # Set up matplotlib
    setup_matplotlib()

    # Perform HuggingFace authentication
    print("🔐 Authenticating with HuggingFace...")
    ensure_hf_authentication()


class PlottingProgress:
    """Context manager for tracking plotting progress across datasets."""

    def __init__(self, total_datasets: int, analysis_name: str):
        self.total_datasets = total_datasets
        self.analysis_name = analysis_name
        self.completed_datasets = 0
        self.total_plots = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def dataset_completed(self, dataset_name: str, plots_created: int = 1):
        """Mark a dataset as completed and update progress."""
        self.completed_datasets += 1
        self.total_plots += plots_created
        print(f"✅ Completed {dataset_name} ({self.completed_datasets}/{self.total_datasets})")

    def dataset_skipped(self, dataset_name: str, reason: str = "no data"):
        """Mark a dataset as skipped."""
        self.completed_datasets += 1
        print(f"⏭️  Skipped {dataset_name} ({reason}) ({self.completed_datasets}/{self.total_datasets})")


def validate_data_availability(data, dataset_name: str, models: List[str],
                               shots: List[int] = None) -> bool:
    """
    Validate that data is available for the specified criteria.
    
    Args:
        data: DataFrame containing evaluation results
        dataset_name: Name of the dataset
        models: List of model names
        shots: Optional list of shot counts to validate
        
    Returns:
        True if data is available, False otherwise
    """
    if data.empty:
        print(f"⚠️  No data found for dataset: {dataset_name} - skipping")
        return False

    # Check if any models have data
    models_with_data = filter_models_with_data(data, models, dataset_name, shots)

    if not models_with_data:
        print(f"⚠️  No models have complete data for dataset {dataset_name} - skipping plot")
        return False

    return True
