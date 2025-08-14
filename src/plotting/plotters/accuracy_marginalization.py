#!/usr/bin/env python3
"""
Improved Accuracy Marginalization Analyzer

Analyzes the accuracy marginalization across different prompt elements.
This script creates detailed plots showing how various prompt components 
(instruction_phrasings, enumerators, separators, choice ordering) affect model accuracy.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend - no display windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.plotting.utils.config import (
    DEFAULT_MODELS, FULL_DATASETS, DEFAULT_SHOTS, DEFAULT_NUM_PROCESSES,
    PLOT_STYLE,
    get_model_display_name, format_dataset_name, get_output_directory
)
from src.plotting.utils.data_manager import DataManager
from src.plotting.utils.auth import ensure_hf_authentication
from src.plotting.utils.config import get_cache_directory


class AccuracyMarginalizationAnalyzer:
    """
    Analyzes the accuracy marginalization across different prompt elements.

    This class creates visualization plots showing how various prompt components
    affect model accuracy across different datasets. It generates separate graphs
    for each dataset and model combination.
    """

    def __init__(self) -> None:
        """Initialize the analyzer with consistent plot styling and element mappings."""
        # Style settings from config file
        plt.rcParams['font.family'] = PLOT_STYLE['font_family']
        plt.rcParams['font.serif'] = PLOT_STYLE['font_serif']
        plt.rcParams['figure.dpi'] = PLOT_STYLE['figure_dpi']

        # Colors for different components
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']

        # Template name mapping (from original file)
        self.instruction_phrasing_mapping = {
            # From BasicMCPrompts.py
            'MultipleChoiceTemplatesInstructionsWithTopic': 'mmlu_paper',
            'MultipleChoiceTemplatesInstructionsWithoutTopic': 'mmlu_paper_without_topic_original',
            'MultipleChoiceTemplatesInstructionsWithoutTopicFixed': 'mmlu_paper_without_topic',
            'MultipleChoiceTemplatesInstructionsWithTopicHelm': 'helm_with_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHelm': 'helm_without_topic_original',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed': 'helm_without_topic',
            'MultipleChoiceTemplatesInstructionsWithoutTopicHarness': 'lm_evaluation_harness',
            'MultipleChoiceTemplatesStructuredWithTopic': 'structured_with_topic',
            'MultipleChoiceTemplatesStructuredWithoutTopic': 'structured_without_topic',
            'MultipleChoiceTemplatesInstructionsProSAAddress': 'paraphrase_1',
            'MultipleChoiceTemplatesInstructionsProSACould': 'paraphrase_2',
            'MultipleChoiceTemplatesInstructionsStateBelow': 'paraphrase_3',
            'MultipleChoiceTemplatesInstructionsProSASimple': 'paraphrase_4',
            'MultipleChoiceTemplatesInstructionsStateHere': 'paraphrase_5',
            'MultipleChoiceTemplatesInstructionsStateBelowPlease': 'paraphrase_6',

            # From SocialQaPrompts.py and OpenBookQAPrompts.py
            'MultipleChoiceTemplatesInstructionsWithTopicAndCoT': 'chain_of_thought',

            # From HellaSwagPrompts.py
            'MultipleChoiceTemplatesInstructionsStandard': 'hellaswag_standard',
            'MultipleChoiceTemplatesInstructionsContext': 'hellaswag_context',
            'MultipleChoiceTemplatesInstructionsStructured': 'hellaswag_structured',
            'MultipleChoiceTemplatesInstructionsBasic': 'hellaswag_basic',
            'MultipleChoiceTemplatesInstructionsState1': 'hellaswag_state_1',
            'MultipleChoiceTemplatesInstructionsState2': 'hellaswag_state_2',
            'MultipleChoiceTemplatesInstructionsState3': 'hellaswag_state_3',
            'MultipleChoiceTemplatesInstructionsState4': 'hellaswag_state_4',
            'MultipleChoiceTemplatesInstructionsState5': 'hellaswag_state_5',
            'MultipleChoiceTemplatesInstructionsState6': 'hellaswag_state_6',
            'MultipleChoiceTemplatesInstructionsState7': 'hellaswag_state_7',
            'MultipleChoiceTemplatesInstructionsState8': 'hellaswag_state_8',

            # From RacePrompts.py
            'MultipleChoiceContextTemplateBasic': 'race_basic',
            'MultipleChoiceContextTemplateBasicNoContextLabel': 'race_basic_no_label',
            'MultipleChoiceContextTemplateMMluStyle': 'race_mmlu_style',
            'MultipleChoiceContextTemplateMMluHelmStyle': 'race_mmlu_helm_style',
            'MultipleChoiceContextTemplateMMluHelmWithChoices': 'race_mmlu_helm_choices',
            'MultipleChoiceContextTemplateProSASimple': 'race_prosa_simple',
            'MultipleChoiceContextTemplateProSACould': 'race_prosa_could',
            'MultipleChoiceContextTemplateStateNumbered': 'race_state_numbered',
            'MultipleChoiceContextTemplateStateOptions': 'race_state_options',
            'MultipleChoiceContextTemplateStateSelect': 'race_state_select',
            'MultipleChoiceContextTemplateStateRead': 'race_state_read',
            'MultipleChoiceContextTemplateStateMultipleChoice': 'race_state_multiple_choice',
        }

        # Enumerator mappings
        self.enumerator_mapping = {
            "capitals": "A,B,C,D",
            "lowercase": "a,b,c,d",
            "numbers": "1,2,3,4",
            "roman": "I,II,III,IV",
            "keyboard": "!,@,#,$",
            "greek": "α,β,γ,δ"
        }

        # Factor display names
        self.factor_display_names = {
            "instruction_phrasing": "Instruction Phrasing",
            "enumerator": "Enumerator",
            "separator": "Choice Separator",
            "choices_order": "Choice Order",
            "prompt_format": "Prompt Format",
            "few_shot_fingerprint": "Few-shot Fingerprint",
            "shots": "Number of Demonstrations"
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data to adapt to expected format.

        Args:
            df: Raw DataFrame from data manager

        Returns:
            Processed DataFrame with renamed columns and mapped values
        """
        processed_df = df.copy()

        # Change column names to format expected by analyzer; create columns if missing
        column_mapping = {
            'dimensions_4: instruction_phrasing_text': 'instruction_phrasing',
            'dimensions_1: enumerator': 'enumerator',
            'dimensions_2: separator': 'separator',
            'dimensions_3: choices_order': 'choices_order',
            'dimensions_5: shots': 'shots',
            'dimensions_6: prompt_format': 'prompt_format',
            'dimensions_7: few_shot_fingerprint': 'few_shot_fingerprint',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in processed_df.columns:
                processed_df[new_col] = processed_df[old_col]
            else:
                processed_df[new_col] = np.nan

        # Shorten model names
        processed_df['model_name'] = processed_df['model'].apply(lambda x: x.split('/')[-1])
        processed_df['model_name'] = processed_df['model_name'].apply(lambda x: x.replace('Meta-', ''))

        # Fix separators if present
        if 'separator' in processed_df.columns:
            processed_df['separator'] = processed_df['separator'].where(
                processed_df['separator'].isna(),
                processed_df['separator'].astype(str).str.replace('\n', '\\n')
            )

        # Map instruction_phrasings
        processed_df['instruction_phrasing'] = processed_df['instruction_phrasing'].map(
            self.instruction_phrasing_mapping).fillna(
            processed_df['instruction_phrasing'])

        # Map enumerators if present
        if 'enumerator' in processed_df.columns:
            processed_df['enumerator'] = processed_df['enumerator'].map(self.enumerator_mapping).fillna(
                processed_df['enumerator'])

        return processed_df

    def _filter_available_factors(self, df: pd.DataFrame, candidate_factors: List[str]) -> List[str]:
        """Return factors that exist and are not all-null or constant within the dataset."""
        available = []
        for factor in candidate_factors:
            if factor not in df.columns:
                continue
            series = df[factor]
            non_null = series.dropna()
            if non_null.empty:
                continue
            if non_null.nunique(dropna=True) <= 1:
                continue
            available.append(factor)
        return available

    def _calculate_factor_widths(self, model_data: pd.DataFrame, factors: List[str]) -> List[float]:
        """Calculate appropriate width for each factor based on number of values."""
        factor_widths = []
        for factor in factors:
            factor_data = model_data.dropna(subset=[factor])
            if not factor_data.empty:
                num_values = len(factor_data[factor].unique())
                # Template usually has many values, so give it extra width
                if factor == 'instruction_phrasing':
                    factor_widths.append(max(6, num_values * 0.8))
                else:
                    factor_widths.append(max(4, num_values * 0.6))
            else:
                factor_widths.append(4)  # Default width
        return factor_widths

    def _get_label_settings(self, factor: str, num_values: int) -> Tuple[int, int, str]:
        """Get label font size, rotation, and alignment based on factor type."""
        if factor in ['instruction_phrasing', 'choices_order', 'prompt_format',
                         'few_shot_fingerprint']:
            # Always rotate instruction_phrasing labels due to long text
            label_fontsize = 8 if num_values > 6 else 9
            label_rotation = 45
            ha_alignment = 'right'
        else:
            # Other factors (enumerator, separator) - no rotation needed
            label_fontsize = 10
            label_rotation = 0
            ha_alignment = 'center'
        return label_fontsize, label_rotation, ha_alignment

    def _shorten_labels(self, factor_values: List[str], factor: str) -> List[str]:
        """Shorten labels if they are too long."""
        short_labels = []
        for val in factor_values:
            if factor == 'instruction_phrasing':
                # For instruction_phrasing, shorten more to save space
                if len(str(val)) > 12:
                    short_labels.append(str(val)[:9] + '...')
                else:
                    short_labels.append(str(val))
            elif len(str(val)) > 15:
                short_labels.append(str(val)[:12] + '...')
            else:
                short_labels.append(str(val))
        return short_labels

    def _plot_factor(self, ax, factor: str, model_data: pd.DataFrame, idx: int) -> None:
        """Plot a single factor on the given axis."""
        ax.set_facecolor('white')

        # Remove frames
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Filter data for this factor
        factor_data = model_data.dropna(subset=[factor])

        if factor_data.empty:
            ax.text(0.5, 0.5, f'No data for {factor}', ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate statistics
        stats = factor_data.groupby(factor)['accuracy'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        stats['mean'] = stats['mean'] * 100  # Convert to percentages

        # Sort by average (high to low)
        stats = stats.sort_values('mean', ascending=False)

        factor_values = stats[factor].tolist()

        if not factor_values:
            ax.text(0.5, 0.5, f'No values for {factor}', ha='center', va='center', transform=ax.transAxes)
            return

        # Fixed color mapping
        color_map = {value: self.colors[i % len(self.colors)] for i, value in enumerate(factor_values)}

        # Adjust bar width based on number of values
        num_values = len(factor_values)
        if factor == 'instruction_phrasing' and num_values > 6:
            bar_width = 0.4
            text_offset = 0.8
        else:
            bar_width = 0.6
            text_offset = 0.5

        # Create bars
        x = np.arange(len(factor_values))
        bars = ax.bar(x, stats['mean'],
                      color=[color_map[val] for val in factor_values],
                      alpha=0.7, width=bar_width)

        # Display numerical values on bars
        for i, (bar, value) in enumerate(zip(bars, stats['mean'])):
            height = bar.get_height()

            # For instruction_phrasing with many values, use smaller font size
            if factor == 'instruction_phrasing' and num_values > 6:
                font_size = 9
                rotation = 0
            else:
                font_size = 11
                rotation = 0

            ax.text(bar.get_x() + bar.get_width() / 2., height + text_offset,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=font_size,
                    fontweight='bold', rotation=rotation)

        # Graph styling
        factor_title = self.factor_display_names.get(factor, factor.replace('_', ' ').title())
        ax.set_title(factor_title, fontsize=14, fontfamily='DejaVu Serif')

        # X-axis labels
        ax.set_xticks(x)
        short_labels = self._shorten_labels(factor_values, factor)
        label_fontsize, label_rotation, ha_alignment = self._get_label_settings(factor, num_values)
        ax.set_xticklabels(short_labels, rotation=label_rotation, ha=ha_alignment, fontsize=label_fontsize)

        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontfamily='DejaVu Serif')

        # Adjust Y boundaries for rotated text
        if factor in ['instruction_phrasing', 'choices_order']:
            max_mean = max(stats['mean']) if len(stats['mean']) > 0 else 0
            ax.set_ylim(0, min(100, max_mean * 1.15))
        else:
            max_mean = max(stats['mean']) if len(stats['mean']) > 0 else 0
            ax.set_ylim(0, min(100, max_mean * 1.1))

        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    def _create_model_plot(
            self,
            model: str,
            model_data: pd.DataFrame,
            dataset_name: str,
            available_factors: List[str],
            title_suffix: str,
            filename: str,
            output_dir: str
    ) -> None:
        """Create a single plot for a model with the given factors."""
        # Calculate axis width based on number of values in each factor
        factor_widths = self._calculate_factor_widths(model_data, available_factors)
        total_width = sum(factor_widths)

        fig, axes = plt.subplots(1, len(available_factors), figsize=(total_width, 6))

        # White background
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)

        if len(available_factors) == 1:
            axes = [axes]

        # Plot each factor
        for idx, (factor, ax) in enumerate(zip(available_factors, axes)):
            self._plot_factor(ax, factor, model_data, idx)

        # General title for model
        model_display_name = get_model_display_name(model).replace('\n', ' ')
        formatted_dataset_name = format_dataset_name(dataset_name)
        fig.suptitle(f'{formatted_dataset_name} - {model_display_name}\n{title_suffix}',
                     fontsize=16, fontfamily='DejaVu Serif', y=0.95)

        # Create hierarchical folder: model/dataset
        model_short_name = model.split('/')[-1]
        safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
        model_dataset_output_dir = f'{output_dir}/{model_short_name}/{safe_dataset_name}'
        Path(model_dataset_output_dir).mkdir(parents=True, exist_ok=True)

        plt.tight_layout()

        # Save in hierarchical folder
        plt.savefig(f'{model_dataset_output_dir}/{filename}.png',
                    dpi=PLOT_STYLE.get('save_dpi', PLOT_STYLE['figure_dpi']),
                    bbox_inches=PLOT_STYLE['bbox_inches'],
                    transparent=PLOT_STYLE['transparent'],
                    facecolor=PLOT_STYLE['facecolor'])

        plt.close()

    def create_analysis_plots(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            factors: Optional[List[str]] = None,
            shots: int = 0,
            output_dir: Path = None
    ) -> None:
        """
        Create accuracy marginalization analysis for one dataset - separate graph for each model.

        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to analyze
            models: List of model names to include
            factors: List of prompt factors to analyze
            shots: Number of shots (0 for zero-shot, 5 for few-shot)
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('accuracy_marginalization')
        
        if factors is None:
            factors = [
                "instruction_phrasing", "enumerator", "separator",
                "choices_order", "prompt_format", "few_shot_fingerprint"
            ]

        # Filter data for specific dataset and shots
        dataset_data = data[(data['dataset'] == dataset_name)].copy()

        # Ensure shots column is normalized for filtering
        if 'shots' not in dataset_data.columns and 'dimensions_5: shots' in dataset_data.columns:
            dataset_data['shots'] = dataset_data['dimensions_5: shots']

        # Apply the shots filter if possible
        if 'shots' in dataset_data.columns:
            dataset_data = dataset_data[dataset_data['shots'] == shots].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name}, shots: {shots} - skipping")
            return

        # Preprocessing
        processed_data = self._preprocess_data(dataset_data)

        # Check which models have data
        models_with_data = []
        for model in models:
            model_short = model.split('/')[-1].replace('Meta-', '')
            if model_short in processed_data['model_name'].values:
                models_with_data.append(model)
            else:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")

        if not models_with_data:
            print(f"⚠️  No models have data for dataset {dataset_name}, shots: {shots} - skipping plot")
            return

        # Filter factors that exist and are informative (not all-null or constant)
        available_factors = self._filter_available_factors(processed_data, factors)

        if not available_factors:
            print(f"⚠️  No valid factors found for dataset {dataset_name} - skipping")
            return

        print(
            f"📊 Creating accuracy marginalization analysis for {dataset_name} with {len(models_with_data)} models and {len(available_factors)} factors")
        print(f"   Will create separate plot for each model")

        shots_text = f"{shots}-shot"

        # Create separate graph for each model
        for model in models_with_data:
            # Check if file already exists for this model
            model_short_name = model.split('/')[-1]
            safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
            filename = f"accuracy_marginalization_{shots}shot"

            output_png = f'{output_dir}/{model_short_name}/{safe_dataset_name}/{filename}.png'

            if os.path.exists(output_png):
                print(f"⏭️  Skipping {model_short_name}/{dataset_name} ({shots}shot) - file already exists")
                continue

            # Filter data for this specific model
            model_short = model.split('/')[-1].replace('Meta-', '')
            model_data = processed_data[processed_data['model_name'] == model_short]

            if model_data.empty:
                print(f"⚠️  No processed data for model {model_short} - skipping")
                continue

            # Create plot
            self._create_model_plot(
                model=model,
                model_data=model_data,
                dataset_name=dataset_name,
                available_factors=available_factors,
                title_suffix=f"Accuracy Marginalization ({shots_text})",
                filename=filename,
                output_dir=output_dir
            )

            model_display_name = get_model_display_name(model).replace('\n', ' ')
            print(
                f"✅ Created accuracy marginalization analysis for {model_display_name} on {dataset_name} ({shots_text})")

        print(
            f"✅ Completed accuracy marginalization analysis for {dataset_name} ({shots_text}) - created {len(models_with_data)} model-specific plots")

    def create_combined_analysis(
            self,
            data: pd.DataFrame,
            dataset_name: str,
            models: List[str],
            factors: Optional[List[str]] = None,
            output_dir: Path = None
    ) -> None:
        """
        Create combined accuracy marginalization analysis (0-shot and 5-shot together) for one dataset.

        This creates a comprehensive analysis that includes shots as an additional dimension,
        showing how accuracy marginalization interacts with few-shot learning.

        Args:
            data: DataFrame containing evaluation results
            dataset_name: Name of the dataset to analyze
            models: List of model names to include
            factors: List of prompt factors to analyze (defaults to 5 including shots)
            output_dir: Directory to save plots (Path object)
        """
        if output_dir is None:
            output_dir = get_output_directory('accuracy_marginalization')
        
        if factors is None:
            factors = [
                "instruction_phrasing", "enumerator", "separator",
                "choices_order", "prompt_format", "few_shot_fingerprint", "shots"
            ]

        # Filter data for specific dataset (all shots)
        dataset_data = data[data['dataset'] == dataset_name].copy()

        if dataset_data.empty:
            print(f"⚠️  No data found for dataset: {dataset_name} - skipping combined analysis")
            return

        # Preprocessing
        processed_data = self._preprocess_data(dataset_data)

        # Add shots as a factor (human-readable for combined plots)
        if 'shots' not in processed_data.columns and 'dimensions_5: shots' in processed_data.columns:
            processed_data['shots'] = processed_data['dimensions_5: shots']
        if 'shots' in processed_data.columns:
            processed_data['shots'] = processed_data['shots'].astype(int)
            processed_data['shots'] = processed_data['shots'].astype(str) + '-shot'

        # Check which models have data
        models_with_data = []
        for model in models:
            model_short = model.split('/')[-1].replace('Meta-', '')
            if model_short in processed_data['model_name'].values:
                models_with_data.append(model)
            else:
                print(f"⚠️  No data for model {model} on dataset {dataset_name} - skipping this model")

        if not models_with_data:
            print(f"⚠️  No models have data for dataset {dataset_name} - skipping combined plot")
            return

        # Filter factors that exist and are informative (not all-null or constant)
        available_factors = self._filter_available_factors(processed_data, factors)

        if not available_factors:
            print(f"⚠️  No valid factors found for dataset {dataset_name} - skipping combined analysis")
            return

        print(
            f"📊 Creating COMBINED accuracy marginalization analysis for {dataset_name} with {len(models_with_data)} models")
        print(f"   Factors: {available_factors}")

        # Create separate combined graph for each model
        for model in models_with_data:
            model_short_name = model.split('/')[-1]
            safe_dataset_name = dataset_name.replace('.', '_').replace('/', '_')
            filename = f"accuracy_marginalization_combined"

            output_png = f'{output_dir}/{model_short_name}/{safe_dataset_name}/{filename}.png'

            if os.path.exists(output_png):
                print(f"⏭️  Skipping {model_short_name}/{dataset_name} (combined) - file already exists")
                continue

            # Filter data for this specific model
            model_short = model.split('/')[-1].replace('Meta-', '')
            model_data = processed_data[processed_data['model_name'] == model_short]

            if model_data.empty:
                print(f"⚠️  No processed data for model {model_short} - skipping combined")
                continue

            # Create plot
            self._create_model_plot(
                model=model,
                model_data=model_data,
                dataset_name=dataset_name,
                available_factors=available_factors,
                title_suffix="Accuracy Marginalization",
                filename=filename,
                output_dir=output_dir
            )

            model_display_name = get_model_display_name(model).replace('\n', ' ')
            print(f"✅ Created COMBINED accuracy marginalization analysis for {model_display_name} on {dataset_name}")

        print(
            f"✅ Completed COMBINED accuracy marginalization analysis for {dataset_name} - created {len(models_with_data)} model-specific plots")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description='Create accuracy marginalization analysis plots for LLM evaluation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_accuracy_marginalization.py
  python run_accuracy_marginalization.py --models meta-llama/Llama-3.2-1B-Instruct
  python run_accuracy_marginalization.py --datasets mmlu.college_biology
        """
    )

    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'List of models to analyze (default: {len(DEFAULT_MODELS)} models)')

    parser.add_argument('--datasets', nargs='+', default=FULL_DATASETS,
                        help=f'List of datasets to analyze (default: {len(FULL_DATASETS)} datasets)')

    parser.add_argument('--factors', nargs='+',
                        default=['instruction_phrasing', 'enumerator', 'separator', 'choices_order', 'prompt_format', 'few_shot_fingerprint'],
                        choices=['instruction_phrasing', 'enumerator', 'separator', 'choices_order', 'prompt_format', 'few_shot_fingerprint'],
                        help='List of prompt factors to analyze (default: instruction_phrasing enumerator separator choices_order)')

    parser.add_argument('--shots', nargs='+', type=int, default=DEFAULT_SHOTS,
                        help=f'List of shot counts to analyze (default: {DEFAULT_SHOTS})')

    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                        help=f'Number of parallel processes (default: {DEFAULT_NUM_PROCESSES})')

    parser.add_argument('--output-dir', type=Path, default=get_output_directory('accuracy_marginalization'),
                        help=f'Output directory for plots (default: {get_output_directory("accuracy_marginalization")})')

    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (default: cache enabled)')

    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing plot files (default: skip existing)')

    parser.add_argument('--list-models', action='store_true',
                        help='List available default models and exit')

    parser.add_argument('--list-datasets', action='store_true',
                        help='List available default datasets and exit')

    return parser.parse_args()


def main() -> None:
    """
    Main function - processes each dataset separately for memory efficiency.

    Creates detailed accuracy marginalization analysis showing how different prompt components
    (instruction_phrasings, enumerators, separators, choice ordering) affect model performance.
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
    factors_to_analyze = args.factors
    shots_to_evaluate = args.shots
    num_processes = args.num_processes
    use_cache = not args.no_cache
    output_dir = args.output_dir
    force_overwrite = args.force

    print("Starting Accuracy Marginalization Analysis")
    print(f"Models: {len(models_to_evaluate)}")
    print(f"Datasets: {len(selected_datasets)}")
    print(f"Factors: {factors_to_analyze}")
    print(f"Shots: {shots_to_evaluate}")
    print(f"Processes: {num_processes}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Force overwrite: {'enabled' if force_overwrite else 'disabled (skip existing)'}")
    print("MEMORY EFFICIENT: Processing one dataset at a time")
    print("=" * 60)

    # Create analyzer
    with DataManager(use_cache=use_cache, persistent_cache_dir=get_cache_directory()) as data_manager:
        analyzer = AccuracyMarginalizationAnalyzer()

        # Process dataset by dataset - memory efficient
        total_plots = 0

        for dataset in tqdm(selected_datasets, desc="Processing datasets"):
            safe_dataset_name = dataset.replace('.', '_').replace('/', '_')
            models_needing_plots = []
            if not force_overwrite:
                for model in models_to_evaluate:
                    model_short_name = model.split('/')[-1]
                    model_dataset_dir = f'{output_dir}/{model_short_name}/{safe_dataset_name}'
                    model_needs_plots = False
                    for shots in shots_to_evaluate:
                        filename = f"accuracy_marginalization_{shots}shot.png"
                        output_file = f'{model_dataset_dir}/{filename}'
                        if not os.path.exists(output_file):
                            model_needs_plots = True
                            break
                    combined_filename = f"accuracy_marginalization_combined.png"
                    combined_output_file = f'{model_dataset_dir}/{combined_filename}'
                    if not model_needs_plots and not os.path.exists(combined_output_file):
                        model_needs_plots = True
                    if model_needs_plots:
                        models_needing_plots.append(model)
            else:
                models_needing_plots = models_to_evaluate
            if not models_needing_plots:
                print(f"⏭️  Skipping {dataset} - all plots already exist for all models")
                continue
            print(f"Loading data for dataset: {dataset}")
            print(f"  Models needing plots: {len(models_needing_plots)}/{len(models_to_evaluate)}")
            for model in models_needing_plots:
                print(f"    - {model.split('/')[-1]}")
            dataset_data = data_manager.load_multiple_models(
                model_names=models_needing_plots,
                datasets=[dataset],
                shots_list=shots_to_evaluate,
                aggregate=True,
                num_processes=num_processes,
                force_reload=force_overwrite
            )
            if dataset_data.empty:
                print(f"⚠️  No data loaded for dataset: {dataset} - skipping")
                continue
            print(f"✓ Loaded data for {dataset}: {dataset_data.shape}")
            for shots in shots_to_evaluate:
                shots_text = "zero-shot" if shots == 0 else f"{shots}-shot"
                print(f"Creating {shots_text} accuracy marginalization analysis for {dataset}...")
                analyzer.create_analysis_plots(
                    data=dataset_data,
                    dataset_name=dataset,
                    models=models_needing_plots,
                    factors=factors_to_analyze,
                    shots=shots,
                    output_dir=output_dir
                )
                total_plots += 1
            print(f"Creating COMBINED accuracy marginalization analysis for {dataset}...")
            analyzer.create_combined_analysis(
                data=dataset_data,
                dataset_name=dataset,
                models=models_needing_plots,
                factors=None,
                output_dir=output_dir
            )
            total_plots += 1
            del dataset_data
            print(f"✅ Completed {dataset} and freed memory")

        print("=" * 60)
        print(f"✓ Accuracy Marginalization Analysis completed!")
        print(f"  - Processed {total_plots} dataset-shot combinations")
        print(f"  - Each dataset processed individually (memory efficient)")
        print(f"  - Created separate plot for each model")
        print(f"  - Each plot shows all factors for that specific model")
        print(f"  - Analyzed factors: {factors_to_analyze}")
        print(f"  - For each dataset: 3 files per model:")
        print(f"    • 0-shot analysis (4 factors): instruction_phrasing, enumerator, separator, choices_order")
        print(f"    • 5-shot analysis (4 factors): instruction_phrasing, enumerator, separator, choices_order")
        print(f"    • Combined analysis (5 factors): instruction_phrasing, enumerator, separator, choices_order, shots")
        print(f"  - Plots saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
