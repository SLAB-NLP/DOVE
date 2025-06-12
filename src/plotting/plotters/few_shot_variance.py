#!/usr/bin/env python3
"""
Improved Few-Shot Variance Analysis Plots

Creates few-shot variance analysis plots for LLM evaluation data.
This script generates separate plots for each dataset showing the variance in performance
between 0-shot and 5-shot prompting across different models.
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for relative imports when running directly
from src.plotting.utils.config import (
    DEFAULT_MODELS, DEFAULT_SHOTS, DEFAULT_DATASETS, DEFAULT_NUM_PROCESSES,
    PLOT_STYLE,
    get_model_display_name, format_dataset_name, get_output_directory
)
from src.plotting.utils.data_manager import DataManager
from src.plotting.utils.config import get_cache_directory
from src.plotting.utils.auth import ensure_hf_authentication


class FewShotVarianceAnalyzer:
    """
    Creates few-shot variance analysis plots using the same style as the original implementation.
    
    This class generates side-by-side box plots with scatter points analyzing the variance
    in performance between 0-shot and 5-shot prompting for each model and dataset combination.
    """

    def __init__(self):
        """Initialize the visualizer with consistent plot styling from config."""
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['mathtext.fontset'] = PLOT_STYLE['mathtext_fontset']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

    def create_comparison_plot(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            output_dir: Path = None
    ):
        """
        Create a few-shot variance analysis plot for a single dataset.
        
        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to plot
            models: List of model names to include
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('few_shot_variance')
        
        # Filter data for the specific dataset
        dataset_data = data[data['dataset'] == dataset_name].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name} - skipping")
            return

        # Check that at least one model has data for both shot types
        models_with_data = []
        for model in models:
            model_data = dataset_data[dataset_data['model'] == model]

            if model_data.empty:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")
                continue

            # Check that there's data for both zero-shot and few-shot
            zero_shot = model_data[model_data['dimensions_5: shots'] == 0]
            five_shot = model_data[model_data['dimensions_5: shots'] == 5]

            models_with_data.append(model)

        if not models_with_data:
            print(f"⚠️  No models have complete variance analysis data for dataset {dataset_name} - skipping plot")
            return

        print(f"📊 Creating few-shot variance plot for {dataset_name} with {len(models_with_data)} models")

        # Create the plot using original style
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # White background instead of transparent
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')

        # Jitter width for scatter points
        jitter_width = 0.2

        # For each model with data
        for i, model in enumerate(models_with_data):
            model_data = dataset_data[dataset_data['model'] == model]

            # Split by zero-shot and few-shot
            zero_shot = model_data[model_data['dimensions_5: shots'] == 0]
            five_shot = model_data[model_data['dimensions_5: shots'] == 5]

            # Accuracy data (convert to percentages)
            zero_values = zero_shot['accuracy'].values * 100
            five_values = five_shot['accuracy'].values * 100

            # Create positions with jitter
            zero_x = np.random.normal(i - jitter_width / 2, jitter_width / 4, size=len(zero_values))
            five_x = np.random.normal(i + jitter_width / 2, jitter_width / 4, size=len(five_values))

            # Plot scatter points with fixed colors
            ax.scatter(zero_x, zero_values, color='red', alpha=0.6, s=50,
                       label='0-shot' if i == 0 else "")
            ax.scatter(five_x, five_values, color='blue', alpha=0.6, s=50,
                       label='5-shot' if i == 0 else "")

            # Add boxplots
            bp_zero = ax.boxplot([zero_values], positions=[i - jitter_width / 2],
                                 widths=jitter_width / 2, patch_artist=True,
                                 medianprops=dict(color='black'),
                                 flierprops=dict(marker='none'),
                                 boxprops=dict(facecolor='red', alpha=0.3),
                                 showfliers=False)

            bp_five = ax.boxplot([five_values], positions=[i + jitter_width / 2],
                                 widths=jitter_width / 2, patch_artist=True,
                                 medianprops=dict(color='black'),
                                 flierprops=dict(marker='none'),
                                 boxprops=dict(facecolor='blue', alpha=0.3),
                                 showfliers=False)

        # Style the plot
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, len(models_with_data) - 0.5)

        # Title with dataset name
        formatted_dataset_name = format_dataset_name(dataset_name)
        ax.set_title(formatted_dataset_name, fontsize=21, fontfamily='DejaVu Serif', pad=15)
        ax.set_ylabel('Accuracy Score (%)', fontsize=18, fontfamily='DejaVu Serif')

        # Model labels on X-axis using short names from config
        model_labels = [get_model_display_name(model) for model in models_with_data]
        ax.set_xticks(range(len(models_with_data)))
        ax.set_xticklabels(model_labels, rotation=0, ha='center',
                           fontsize=14, fontfamily='DejaVu Serif')

        # Tick settings
        ax.tick_params(axis='y', labelsize=16)

        # Legend matching original style
        ax.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, 1.35),
                  ncol=2, frameon=True, edgecolor='black', markerscale=3.0)

        # Create directory by dataset (instead of by models)
        dataset_output_dir = f'{output_dir}/{dataset_name.replace(".", "_").replace("/", "_")}'
        Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

        plt.tight_layout()

        # Save in dataset directory
        plt.savefig(f'{dataset_output_dir}/few_shot_variance_analysis.png',
                    dpi=PLOT_STYLE['figure_dpi'],
                    bbox_inches=PLOT_STYLE['bbox_inches'],
                    transparent=PLOT_STYLE['transparent'],
                    facecolor=PLOT_STYLE['facecolor'])
        plt.savefig(f'{dataset_output_dir}/few_shot_variance_analysis.svg',
                    bbox_inches=PLOT_STYLE['bbox_inches'],
                    transparent=PLOT_STYLE['transparent'],
                    facecolor=PLOT_STYLE['facecolor'])

        plt.close()
        print(f"✅ Created few-shot variance plot for {dataset_name} with {len(models_with_data)} models")


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create few-shot variance analysis plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_few_shot_variance.py
python run_few_shot_variance.py --models meta-llama/Llama-3.2-1B-Instruct
python run_few_shot_variance.py --datasets ai2_arc.arc_challenge hellaswag
python run_few_shot_variance.py --num-processes 8 --no-cache
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'List of datasets to analyze (default: {len(DEFAULT_DATASETS)} datasets)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help=f'Number of parallel processes (default: {DEFAULT_NUM_PROCESSES})')

    parser.add_argument('--output-dir', type=Path, default=get_output_directory('few_shot_variance'),
                        help=f'Output directory for plots (default: {get_output_directory("few_shot_variance")})')

    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (default: cache enabled)')

    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing plot files (default: skip existing)')

    parser.add_argument('--list-models', action='store_true',
                        help='List available default models and exit')

    parser.add_argument('--list-datasets', action='store_true',
                        help='List available default datasets and exit')

    return parser.parse_args()


def main():
    """
    Main function - processes each dataset separately for memory efficiency.
    
    Creates side-by-side variance analysis plots showing 0-shot vs 5-shot performance
    for each dataset with all specified models.
    """
    args = parse_arguments()

    # Handle list options
    if args.list_models:
        print("Default models:")
        for i, model in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {model}")
        return

    if args.list_datasets:
        print("Default datasets for few-shot variance analysis:")
        for i, dataset in enumerate(DEFAULT_DATASETS, 1):
            print(f"  {i}. {dataset}")
        return

    # Perform HuggingFace authentication once at the start
    print("🔐 Authenticating with HuggingFace...")
    ensure_hf_authentication()

    models_to_evaluate = args.models
    selected_datasets = args.datasets
    shots_to_evaluate = args.shots
    num_processes = args.num_processes
    use_cache = not args.no_cache
    output_dir = args.output_dir
    force_overwrite = args.force

    print("Starting Few-Shot Variance Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("MEMORY EFFICIENT: Processing one dataset at a time")
    print("=" * 60)

    # Create data manager and plot generator
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        visualizer = FewShotVarianceAnalyzer()

        # Process dataset by dataset - memory efficient
        total_plots = 0
        print("Processing datasets one by one (memory efficient)...")

        for dataset in tqdm(selected_datasets, desc="Processing datasets"):
            safe_dataset_name = dataset.replace('.', '_').replace('/', '_')
            filename = f"few_shot_variance_analysis"
            dataset_output_dir = f'{output_dir}/{safe_dataset_name}'
            output_png = f'{dataset_output_dir}/{filename}.png'
            if not force_overwrite and os.path.exists(output_png):
                print(f"⏭️  Skipping {dataset} (few-shot variance) - file already exists: {output_png}")
                continue
            print(f"Loading data for dataset: {dataset}")
            dataset_data = data_manager.load_multiple_models(
                model_names=models_to_evaluate,
                datasets=[dataset],
                shots_list=shots_to_evaluate,
                aggregate=True,
                num_processes=num_processes
            )
            if dataset_data.empty:
                print(f"⚠️  No data loaded for dataset: {dataset} - skipping")
                continue
            print(f"✓ Loaded data for {dataset}: {dataset_data.shape}")
            visualizer.create_comparison_plot(
                data=dataset_data,
                dataset_name=dataset,
                models=models_to_evaluate,
                output_dir=output_dir
            )
            total_plots += 1
            del dataset_data
            print(f"✅ Completed {dataset} and freed memory")

        print("=" * 60)
        print(f"✓ Few-Shot Variance Analysis completed!")
        print(f"  - Created {total_plots} separate plots")
        print(f"  - Each dataset processed individually (memory efficient)")
        print(f"  - Plots saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
