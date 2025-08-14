#!/usr/bin/env python3
"""
Improved Performance Variations Plots

Creates performance variation plots for LLM evaluation data with separate plots for each dataset.
This script generates two types of plots:
1. Combined plots showing different shot counts with distinction
2. Unified plots combining all shot counts without distinction
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.plotting.utils.data_manager import DataManager
from src.plotting.utils.config import (
    DEFAULT_MODELS, FULL_DATASETS, DEFAULT_SHOTS, DEFAULT_NUM_PROCESSES,
    PLOT_STYLE,
    get_model_display_name, get_model_color, format_dataset_name, get_output_directory
)
from src.plotting.utils.auth import ensure_hf_authentication
from src.plotting.utils.config import get_cache_directory


class PerformanceVariationsAnalyzer:
    """
    Creates performance variation plots using the same style as the original implementation.
    
    This class generates box plots with scatter points showing the distribution of accuracy
    scores across different prompt configurations for each model and dataset combination.
    """

    def __init__(self):
        """Initialize the visualizer with consistent plot styling from config."""
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

    def _create_lighter_color(self, hex_color: str) -> str:
        """
        Create a lighter version of the given hex color.
        
        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            
        Returns:
            Lighter hex color string
        """
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        lighter_rgb = tuple(int(c + (255 - c) * 0.35) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*lighter_rgb)

    def _create_darker_color(self, hex_color: str) -> str:
        """
        Create a darker version of the given hex color.
        
        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            
        Returns:
            Darker hex color string
        """
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(int(c * 0.7) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darker_rgb)

    def _lighten_color(self, hex_color: str, amount: float) -> str:
        """
        Lighten a hex color by a given amount in [0, 1].
        amount=0 returns the original color; amount=1 returns white.
        """
        amount = max(0.0, min(1.0, amount))
        rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        light_rgb = tuple(int(c + (255 - c) * amount) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*light_rgb)

    def create_performance_plot(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            shots_list: List[int] = [0, 5],
            output_dir: Path = None
    ):
        """
        Create a performance variation plot for a single dataset showing all shot counts together.
        
        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to plot
            models: List of model names to include
            shots_list: List of shot counts to analyze
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('performance_variations')

        # Filter data for the specific dataset (all shots)
        dataset_data = data[data['dataset'] == dataset_name].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name} - skipping")
            return

        # Check that at least one model has data
        models_with_data = []
        for model in models:
            # Support both full model name and short name
            model_short = model.split('/')[-1]
            model_data = dataset_data[
                (dataset_data['model'] == model) |
                (dataset_data['model'] == model_short)
                ]
            if model_data.empty:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")
                continue

            # Check that there's data for at least one shot count
            available_shots = set(model_data['dimensions_5: shots'].unique())
            required_shots = set(shots_list)

            if not available_shots.intersection(required_shots):
                print(
                    f"⚠️  Model {model} has no data for any required shots {shots_list} for {dataset_name} - skipping this model")
                continue

            # Print warning if some shots are missing but still have partial data
            missing_shots = required_shots - available_shots
            if missing_shots:
                print(
                    f"ℹ️  Model {model} missing shots {missing_shots} for {dataset_name} - will show available shots only")

            models_with_data.append(model)

        if not models_with_data:
            print(f"⚠️  No models have complete data for dataset {dataset_name} - skipping plot")
            return

        print(f"📊 Creating combined plot for {dataset_name} with {len(models_with_data)} models and shots {shots_list}")

        # Create the plot - adapted for multiple shot counts
        fig, ax = plt.subplots(figsize=(14, 5.5))

        # White background instead of transparent
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')

        # Remove plot borders for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Layout constants - adapted for multiple shot counts
        model_group_width = 0.4  # Width of model group (for all shots)
        shot_width = 0.15  # Width per shot
        model_spacing = 0.1  # Spacing between models
        shot_spacing = 0.02  # Spacing between shots of the same model

        # Determine which shot counts exist for this dataset intersected with requested shots
        available_dataset_shots = list(sorted(set(dataset_data['dimensions_5: shots'].unique())))
        shots_available = [s for s in shots_list if s in available_dataset_shots]
        max_index = max(0, len(shots_available) - 1)

        # For each model with data
        for model_idx, model in enumerate(models_with_data):
            # Support both full model name and short name
            model_short = model.split('/')[-1]
            model_data = dataset_data[
                (dataset_data['model'] == model) |
                (dataset_data['model'] == model_short)
                ]

            # Calculate base position for this model
            model_center = model_idx * (model_group_width + model_spacing)

            # Colors from config file
            base_color = get_model_color(model)

            # Determine which shots actually have data for this model
            model_shots_present = [
                shots for shots in shots_list
                if not model_data[model_data['dimensions_5: shots'] == shots].empty
            ]

            # If no shots have data for this model, skip plotting this model
            if not model_shots_present:
                continue

            # For each shot count that has data
            for shot_idx, shots in enumerate(model_shots_present):
                shot_data = model_data[model_data['dimensions_5: shots'] == shots]

                if shot_data.empty:
                    continue

                # Calculate position for this shot count
                shots_offset = (shot_idx - (len(model_shots_present) - 1) / 2) * (shot_width + shot_spacing)
                shot_center = model_center + shots_offset

                # Colors by shot count: lighter for fewer shots, consistent across models
                if max_index == 0:
                    lighten_amount = 0.0
                else:
                    shot_rank = shots_available.index(shots)
                    lighten_amount = 0.35 * (max_index - shot_rank) / max_index
                color = self._lighten_color(base_color, lighten_amount)

                # Accuracy data
                accuracies = shot_data['accuracy'].values * 100

                # Jitter for scatter points
                x = np.random.normal(shot_center, shot_width * 0.15, size=len(accuracies))

                # Plot scatter points (legend handled separately)
                ax.scatter(x, accuracies,
                           alpha=0.6,
                           s=20,
                           c=[color],
                           zorder=2)

                # Add boxplot
                bp = ax.boxplot([accuracies],
                                positions=[shot_center],
                                widths=shot_width * 0.7,
                                patch_artist=True,
                                medianprops=dict(color='black', linewidth=1.5),
                                flierprops=dict(marker='none'),
                                boxprops=dict(facecolor=color,
                                              color=self._create_darker_color(color),
                                              alpha=0.7,
                                              linewidth=1.2),
                                whiskerprops=dict(color=self._create_darker_color(color), linewidth=1.2),
                                capprops=dict(color=self._create_darker_color(color), linewidth=1.2),
                                showfliers=False)

        # Style the plot
        ax.set_ylabel('Accuracy Score (%)', fontsize=20, fontfamily='DejaVu Serif', labelpad=15)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.3, len(models_with_data) * (model_group_width + model_spacing) - model_spacing + 0.3)

        # X-axis labels - short names from config (centered on each model)
        model_labels = [get_model_display_name(model) for model in models_with_data]
        model_positions = [i * (model_group_width + model_spacing) for i in range(len(models_with_data))]

        plt.xticks(model_positions, model_labels,
                   rotation=0, ha='center',
                   fontsize=14, fontfamily='DejaVu Serif')
        plt.yticks(fontsize=16, fontfamily='DejaVu Serif')

        # Title
        formatted_dataset_name = format_dataset_name(dataset_name)
        plt.title(f'{formatted_dataset_name}\nPerformance Variations',
                  fontsize=18, fontfamily='DejaVu Serif', pad=20)

        # Legend: one entry per shot count with neutral shades, labeled with exact shot count
        legend_elements = []
        neutral_base = '#808080'
        for idx, shots in enumerate(shots_available):
            if max_index == 0:
                lighten_amount = 0.0
            else:
                lighten_amount = 0.35 * (max_index - idx) / max_index
            shot_color = self._lighten_color(neutral_base, lighten_amount)
            legend_elements.append(
                Patch(facecolor=shot_color,
                      edgecolor=self._create_darker_color(shot_color),
                      label=f'{shots}-shot')
            )
        if legend_elements:
            ax.legend(handles=legend_elements, fontsize=14, loc='upper right', frameon=True,
                      fancybox=True, shadow=True, ncol=1)

        # Create directory by dataset
        safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
        dataset_output_dir = f'{output_dir}/{safe_dataset_name}'
        Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

        # Save in dataset directory
        plt.savefig(f'{dataset_output_dir}/performance_variations.png',
                    dpi=PLOT_STYLE['save_dpi'],
                    bbox_inches=PLOT_STYLE['bbox_inches'],
                    transparent=PLOT_STYLE['transparent'],
                    facecolor=PLOT_STYLE['facecolor'])
        # plt.savefig(f'{dataset_output_dir}/performance_variations.pdf',
        #             bbox_inches=PLOT_STYLE['bbox_inches'],
        #             transparent=PLOT_STYLE['transparent'],
        #             facecolor=PLOT_STYLE['facecolor'])

        plt.close()
        print(
            f"✅ Created combined performance variation plot for {dataset_name} with {len(models_with_data)} models and shots {shots_list}")

    def create_unified_plot(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            shots_list: List[int] = [0, 5],
            output_dir: Path = None
    ):
        """
        Create a unified performance variation plot for a single dataset combining all shot counts.
        
        This version combines all shot counts into a single distribution without distinction.
        
        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to plot
            models: List of model names to include
            shots_list: List of shot counts to combine
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('performance_variations')

        # Filter data for the specific dataset (all shots)
        dataset_data = data[data['dataset'] == dataset_name].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name} - skipping unified plot")
            return

        # Check that at least one model has data
        models_with_data = []
        for model in models:
            # Support both full model name and short name
            model_short = model.split('/')[-1]
            model_data = dataset_data[
                (dataset_data['model'] == model) |
                (dataset_data['model'] == model_short)
                ]
            if model_data.empty:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")
                continue

            # Check that there's data for at least one shot count
            available_shots = set(model_data['dimensions_5: shots'].unique())
            required_shots = set(shots_list)

            if not available_shots.intersection(required_shots):
                print(
                    f"⚠️  Model {model} has no data for any required shots {shots_list} for {dataset_name} - skipping this model")
                continue

            # Print warning if some shots are missing but still have partial data
            missing_shots = required_shots - available_shots
            if missing_shots:
                print(
                    f"ℹ️  Model {model} missing shots {missing_shots} for {dataset_name} - will show available shots only (unified)")

            models_with_data.append(model)

        if not models_with_data:
            print(f"⚠️  No models have complete data for dataset {dataset_name} - skipping unified plot")
            return

        print(
            f"📊 Creating unified plot for {dataset_name} with {len(models_with_data)} models (combining shots {shots_list})")

        # Create the plot - same size but with larger spacing between models
        fig, ax = plt.subplots(figsize=(12, 5.2))

        # White background instead of transparent
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')

        # Remove plot borders for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Layout constants - larger spacing to prevent overlap in unified plot
        model_group_width = 0.2  # Reduced group width
        shot_width = 0.12  # Reduced shot width
        model_spacing = 0.25  # Increased spacing between models

        # For each model with data
        for model_idx, model in enumerate(models_with_data):
            # Support both full model name and short name
            model_short = model.split('/')[-1]
            model_data = dataset_data[
                (dataset_data['model'] == model) |
                (dataset_data['model'] == model_short)
                ]

            # Combine all model data (all shots together)
            all_model_data = []
            for shots in shots_list:
                shot_data = model_data[model_data['dimensions_5: shots'] == shots]
                if not shot_data.empty:
                    all_model_data.extend(shot_data['accuracy'].values)

            if not all_model_data:
                continue

            # Calculate position for this model
            model_center = model_idx * (model_group_width + model_spacing)

            # Colors from config file
            base_color = get_model_color(model)
            color = self._create_lighter_color(base_color)

            # Combined accuracy data (all shots together)
            accuracies = np.array(all_model_data) * 100

            # Jitter for scatter points
            x = np.random.normal(model_center, shot_width * 0.2, size=len(accuracies))

            # Plot scatter points
            ax.scatter(x, accuracies,
                       alpha=0.6,
                       s=16,
                       c=[color],
                       zorder=2)

            # Add boxplot
            bp = ax.boxplot([accuracies],
                            positions=[model_center],
                            widths=shot_width * 0.8,
                            patch_artist=True,
                            medianprops=dict(color='black'),
                            flierprops=dict(marker='none'),
                            boxprops=dict(facecolor=color,
                                          color=self._create_darker_color(color),
                                          alpha=0.7,
                                          linewidth=1.2),
                            whiskerprops=dict(color=self._create_darker_color(color)),
                            capprops=dict(color=self._create_darker_color(color)),
                            showfliers=False)

        # Style the plot
        ax.set_ylabel('Accuracy Score (%)', fontsize=20, fontfamily='DejaVu Serif', labelpad=15)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.3, len(models_with_data) * (model_group_width + model_spacing) - model_spacing + 0.3)

        # X-axis labels - short names from config
        model_labels = [get_model_display_name(model) for model in models_with_data]
        model_positions = [i * (model_group_width + model_spacing) for i in range(len(models_with_data))]

        plt.xticks(model_positions, model_labels,
                   rotation=0, ha='center',
                   fontsize=14, fontfamily='DejaVu Serif')
        plt.yticks(fontsize=16, fontfamily='DejaVu Serif')

        # Title
        formatted_dataset_name = format_dataset_name(dataset_name)
        plt.title(f'{formatted_dataset_name}\nPerformance Variations (Unified)',
                  fontsize=18, fontfamily='DejaVu Serif', pad=20)

        # Create directory by dataset
        safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
        dataset_output_dir = f'{output_dir}/{safe_dataset_name}'
        Path(dataset_output_dir).mkdir(parents=True, exist_ok=True)

        # Save in dataset directory
        plt.savefig(f'{dataset_output_dir}/performance_variations_unified.png',
                    dpi=PLOT_STYLE['save_dpi'],
                    bbox_inches=PLOT_STYLE['bbox_inches'],
                    transparent=PLOT_STYLE['transparent'],
                    facecolor=PLOT_STYLE['facecolor'])

        plt.close()
        print(
            f"✅ Created unified performance variation plot for {dataset_name} with {len(models_with_data)} models (combined shots {shots_list})")


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create performance variation plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_performance_analysis.py
  python run_performance_analysis.py --models meta-llama/Llama-3.2-1B-Instruct
  python run_performance_analysis.py --datasets ai2_arc.arc_challenge hellaswag
  python run_performance_analysis.py --num-processes 8 --no-cache
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=FULL_DATASETS,
                        help=f'List of datasets to analyze (default: {len(FULL_DATASETS)} datasets)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help=f'Number of parallel processes (default: {DEFAULT_NUM_PROCESSES})')

    parser.add_argument('--output-dir', type=Path, default=get_output_directory('performance_variations'),
                        help=f'Output directory for plots (default: {get_output_directory("performance_variations")})')

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
    
    Creates two versions per dataset:
    1. Combined plot with distinction between shot counts
    2. Unified plot without distinction between shot counts
    """
    args = parse_arguments()

    # Handle list options
    if args.list_models:
        print("Default models:")
        for i, model in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {model}")
        return

    if args.list_datasets:
        print("Default datasets:")
        for i, dataset in enumerate(FULL_DATASETS, 1):
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

    print("Starting Performance Variations Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("MEMORY EFFICIENT: Processing one dataset at a time")
    print("Will create 2 versions per dataset: combined (with distinction) + unified (without distinction)")
    print("=" * 60)

    # Create data manager and plot generator
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        visualizer = PerformanceVariationsAnalyzer()

        # Process dataset by dataset - memory efficient
        total_plots = 0
        print(f"Processing datasets one by one (memory efficient)...")
        print("Will create 2 versions per dataset: combined (with distinction) + unified (without distinction)")

        for dataset in tqdm(selected_datasets, desc="Processing datasets"):
            safe_dataset_name = dataset.replace('.', '_').replace('/', '_')
            dataset_output_dir = f'{output_dir}/{safe_dataset_name}'
            output_png1 = f'{dataset_output_dir}/performance_variations.png'
            output_png2 = f'{dataset_output_dir}/performance_variations_unified.png'
            if not force_overwrite and os.path.exists(output_png1) and os.path.exists(output_png2):
                print(f"⏭️  Skipping {dataset} - both files already exist")
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
            if not os.path.exists(output_png1) or force_overwrite:
                visualizer.create_performance_plot(
                    data=dataset_data,
                    dataset_name=dataset,
                    models=models_to_evaluate,
                    shots_list=shots_to_evaluate,
                    output_dir=output_dir,
                )
            else:
                print(f"⏭️  Skipping combined plot for {dataset} - file exists")
            if not os.path.exists(output_png2) or force_overwrite:
                visualizer.create_unified_plot(
                    data=dataset_data,
                    dataset_name=dataset,
                    models=models_to_evaluate,
                    shots_list=shots_to_evaluate,
                    output_dir=output_dir,
                )
            else:
                print(f"⏭️  Skipping unified plot for {dataset} - file exists")
            total_plots += 2
            del dataset_data
            print(f"✅ Completed {dataset} and freed memory")

        print("=" * 60)
        print(f"✓ Performance Variations Analysis completed!")
        print(f"  - Created {total_plots} plots")
        print(f"  - Each dataset processed individually (memory efficient)")
        print(f"  - Plots saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
