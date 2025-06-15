#!/usr/bin/env python3
"""
Improved Success Rate Distribution Analysis

Creates success rate distribution analysis showing how consistently models answer
individual questions correctly across different prompt configurations.
This analysis helps identify the distribution of success rates across questions
and their sensitivity to prompt formatting.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.plotting.utils.config import (
    DEFAULT_MODELS, DEFAULT_DATASETS, DEFAULT_SHOTS, PLOT_STYLE,
    get_model_display_name, get_output_directory
)
from src.plotting.utils.data_manager import DataManager
from src.plotting.utils.auth import ensure_hf_authentication
from src.plotting.utils.config import get_cache_directory


class SuccessRateDistributionAnalyzer:
    """
    Creates success rate distribution analysis plots showing success rate distributions.
    
    This class generates histograms showing how many questions have different levels
    of success rates (consistency) across various prompt configurations.
    """

    def __init__(self):
        """Initialize the analyzer with consistent plot styling from config."""
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

    def _aggregate_samples_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate success rate statistics: how many prompt configurations answered correctly,
        total attempts, and success percentage.
        
        Args:
            df: DataFrame with raw evaluation data
            
        Returns:
            DataFrame with columns:
            - dataset
            - sample_index  
            - sum_correct
            - total_configurations
            - fraction_correct
        """
        print("  Computing success rate distribution statistics...")
        start_time = time.time()

        # Group by dataset and question index - working with 'score' instead of 'accuracy'
        grouped = df.groupby(["dataset", "sample_index"])["score"]

        # Count correct answers and total attempts
        sum_correct = grouped.sum()
        total_configurations = grouped.count()

        # Create DataFrame with statistics
        question_stats = pd.DataFrame({
            "sum_correct": sum_correct,
            "total_configurations": total_configurations
        }).reset_index()

        # Calculate success percentage
        question_stats["fraction_correct"] = (
                question_stats["sum_correct"] / question_stats["total_configurations"]
        ).round(3)

        print(f"  ✓ Aggregation completed in {time.time() - start_time:.2f} seconds")
        return question_stats

    def _create_success_rate_histogram(
            self,
            question_stats: pd.DataFrame,
            model_name: str,
            output_dir: Path = None
    ):
        """
        Create success rate distribution histogram showing success rate distribution across all questions.
        
        Args:
            question_stats: DataFrame with success rate statistics
            model_name: Name of the model being analyzed
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('success_rate_distribution')
        
        if question_stats.empty:
            print(f"⚠️  No question statistics for {model_name} - skipping plot")
            return

        print(f"📊 Creating success rate distribution histogram for {model_name}")
        print(f"  Total questions: {len(question_stats)}")

        # File settings
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"success_rate_distribution"
        model_output_dir = f'{output_dir}/{safe_model_name}'
        output_png = f'{model_output_dir}/{filename}.png'

        # Create the plot - exactly like the original file
        fig = plt.figure(figsize=(18, 12.5))
        ax = plt.gca()

        # White background (not transparent) and remove frames
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Convert to percentages
        percentage_correct = question_stats['fraction_correct'] * 100
        bins = np.arange(0, 105, 5)  # Up to 105 to include 100 in the last bin

        # Create histogram
        n, bins, patches = plt.hist(
            percentage_correct,
            bins=bins,
            edgecolor='black',
            color='skyblue'
        )

        # Color the bars by values - like in the original file
        for i in range(len(patches)):
            bin_center = (bins[i] + bins[i + 1]) / 2
            if bin_center >= 90:
                patches[i].set_facecolor('green')
            elif bin_center <= 10:
                patches[i].set_facecolor('red')
            else:
                patches[i].set_facecolor('skyblue')

        # Add number labels above each bar
        for i in range(len(patches)):
            plt.text(
                patches[i].get_x() + patches[i].get_width() / 2,
                patches[i].get_height(),
                f'{int(n[i])}',
                ha='center',
                va='bottom',
                color='darkblue',
                fontsize=18,
                fontweight='bold'
            )

        # Axis labels
        plt.xlabel('Success Rate Over Different Prompts (%)', fontsize=32)
        plt.ylabel('Number of Questions in Dataset', fontsize=32)

        # Set X-axis range
        plt.xlim(0, 100)

        # Increase font size for labels
        plt.xticks(fontsize=24)
        plt.yticks([])  # Remove Y-axis labels

        # Calculate total number of questions (sum of all bars)
        total_questions = len(question_stats)

        # Title with model name and total question count
        model_display = get_model_display_name(model_name).replace('\n', '')
        plt.title(f'{model_display}\nSuccess Rate Distribution Analysis\n(Total Questions: {total_questions})',
                  fontsize=28, fontfamily='DejaVu Serif', pad=20)

        plt.tight_layout()

        # Create directory for model
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

        # Save
        plt.savefig(
            output_png,
            dpi=PLOT_STYLE['figure_dpi'],
            bbox_inches=PLOT_STYLE['bbox_inches'],
            transparent=PLOT_STYLE['transparent'],
            facecolor=PLOT_STYLE['facecolor']
        )
        plt.savefig(
            f'{model_output_dir}/{filename}.svg',
            bbox_inches=PLOT_STYLE['bbox_inches'],
            transparent=PLOT_STYLE['transparent'],
            facecolor=PLOT_STYLE['facecolor']
        )

        plt.close()

        # Statistics
        total_questions = len(question_stats)
        high_success = len(question_stats[question_stats['fraction_correct'] >= 0.9])
        low_success = len(question_stats[question_stats['fraction_correct'] <= 0.1])

        print(f"✅ Created success rate distribution for {model_name}")
        print(f"  - Total questions: {total_questions}")
        print(f"  - High success rate (≥90%): {high_success} ({high_success / total_questions * 100:.1f}%)")
        print(f"  - Low success rate (≤10%): {low_success} ({low_success / total_questions * 100:.1f}%)")

    def create_success_rate_analysis(
            self,
            model_name: str,
            datasets: List[str],
            shots_list: List[int] = [0, 5],
            data_manager: DataManager = None,
            output_dir: Path = None
    ):
        """
        Perform success rate distribution analysis for each question for one model across all datasets.
        
        Args:
            model_name: Name of the model to analyze
            datasets: List of datasets to include in analysis
            shots_list: List of shot counts to analyze
            data_manager: DataManager instance for loading data
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('success_rate_distribution')
        
        print(f"🔍 Starting success rate distribution analysis for {model_name}")
        print(f"  Datasets: {len(datasets)}")
        print(f"  Shots: {shots_list}")

        all_question_stats = []

        # Process each dataset separately (for memory efficiency)
        for dataset in tqdm(datasets, desc=f"Processing datasets for {model_name}"):
            try:
                # Load raw data for this dataset only - not aggregated!
                dataset_data = data_manager.load_multiple_models(
                    model_names=[model_name],
                    datasets=[dataset],
                    shots_list=shots_list,
                    aggregate=False,  # Important change - raw data!
                    num_processes=1  # No need for parallelism here
                )

                if dataset_data.empty:
                    print(f"⚠️  No data for {model_name} on {dataset} - skipping")
                    continue

                # Add sample_index if not present
                if 'sample_index' not in dataset_data.columns:
                    dataset_data = dataset_data.reset_index()
                    if 'sample_index' not in dataset_data.columns:
                        dataset_data['sample_index'] = range(len(dataset_data))

                # Compute success rate statistics
                dataset_question_stats = self._aggregate_samples_accuracy(dataset_data)

                if not dataset_question_stats.empty:
                    all_question_stats.append(dataset_question_stats)
                    print(f"  ✓ {dataset}: {len(dataset_question_stats)} questions processed")

            except Exception as e:
                print(f"⚠️  Error processing {dataset} for {model_name}: {e}")
                continue

        if not all_question_stats:
            print(f"❌ No valid question statistics found for {model_name}")
            return

        # Combine all statistics from all datasets
        combined_question_stats = pd.concat(all_question_stats, ignore_index=True)

        # Filter questions with minimum number of configurations (like in original file)
        min_configurations = 99  # Can be adjusted
        valid_questions = combined_question_stats[
            combined_question_stats['total_configurations'] >= min_configurations
            ]

        print(f"  📊 Final statistics:")
        print(f"    - Total questions before filtering: {len(combined_question_stats)}")
        print(f"    - Valid questions (≥{min_configurations} configs): {len(valid_questions)}")

        if valid_questions.empty:
            print(f"❌ No questions with sufficient configurations for {model_name}")
            return

        # Create the plot
        self._create_success_rate_histogram(
            question_stats=valid_questions,
            model_name=model_name,
            output_dir=output_dir
        )


def parse_arguments():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create success rate distribution analysis plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_success_rate_distribution.py
python run_success_rate_distribution.py --models meta-llama/Llama-3.2-1B-Instruct
python run_success_rate_distribution.py --datasets ai2_arc.arc_challenge hellaswag
python run_success_rate_distribution.py --num-processes 1 --no-cache
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'List of datasets to analyze (default: {len(DEFAULT_DATASETS)} datasets)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=1,
                        help='Number of parallel processes (default: 1 - sequential processing)')

    parser.add_argument('--output-dir', type=Path, default=get_output_directory('success_rate_distribution'),
                        help=f'Output directory for plots (default: {get_output_directory("success_rate_distribution")})')

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
    Main function - runs the analysis for all models.
    
    Creates success rate distribution histograms showing how consistently each model
    answers individual questions correctly across different prompt configurations.
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

    print("Starting Success Rate Distribution Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("Will create histogram for each model showing question-level success rates across all datasets")
    print("=" * 80)

    # Create analyzer and perform analysis
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        analyzer = SuccessRateDistributionAnalyzer()

        # Analysis for each model separately
        total_analyses = 0

        for model in tqdm(models_to_evaluate, desc="Processing models"):
            safe_model_name = model.replace('/', '_').replace('-', '_')
            filename = f"success_rate_distribution"
            model_output_dir = f'{output_dir}/{safe_model_name}'
            output_png = f'{model_output_dir}/{filename}.png'
            if not force_overwrite and os.path.exists(output_png):
                print(f"⏭️  Skipping {model} - file already exists: {output_png}")
                continue
            print(f"\n{'=' * 60}")
            print(f"Processing model: {model}")
            print(f"{'=' * 60}")
            analyzer.create_success_rate_analysis(
                model_name=model,
                datasets=selected_datasets,
                shots_list=shots_to_evaluate,
                data_manager=data_manager,
                output_dir=output_dir
            )
            total_analyses += 1

        print("\n" + "=" * 80)
        print(f"✓ Success Rate Distribution Analysis completed!")
        print(f"  - Analyzed {total_analyses} models")
        print(f"  - Plots saved to: {output_dir}")
        print(f"  - Each model has its own directory with histogram")
        print("=" * 80)


if __name__ == "__main__":
    main()
