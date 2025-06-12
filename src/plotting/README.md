# LLM Evaluation Plotting System

A comprehensive, modular plotting system for analyzing LLM evaluation results from the DOVE dataset. This system provides professional-grade visualization tools for understanding model performance across different prompting strategies.

## Features

- **Performance Analysis**: Box plots with scatter points showing accuracy distribution across prompt variations
- **Few-Shot Comparison**: Side-by-side 0-shot vs 5-shot performance analysis
- **Prompt Impact Analysis**: Detailed breakdown of how prompt elements affect model performance
- **Robustness Analysis**: Per-question consistency analysis across prompt configurations
- **Memory Efficient**: Processes datasets individually to minimize memory usage
- **Professional Output**: Publication-ready plots in PNG and SVG formats

## Installation

### Quick Setup (5 minutes)

```bash
# Clone and navigate to the plotting system
cd src/analysis/plotting

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace authentication (choose one method):
# Method 1: Environment variable
export HF_TOKEN="your_token_here"

# Method 2: HuggingFace CLI login
huggingface-cli login

# Method 3: Manual token file
echo "your_token_here" > ~/.huggingface/token
```

### Requirements

- Python 3.8+
- HuggingFace account with access token
- ~10GB disk space for caching (optional but recommended)

## Usage

### Quick Start

Run all analyses with default settings:

```bash
# Performance analysis
python run_performance_analysis.py

# Few-shot comparison  
python run_few_shot_comparison.py

# Prompt impact analysis
python run_prompt_impact_analysis.py

# Robustness analysis
python run_robustness_analysis.py
```

### Advanced Usage

#### Performance Analysis
```bash
python run_performance_analysis.py \
    --models meta-llama/Llama-3.2-1B-Instruct mistralai/Mistral-7B-Instruct-v0.3 \
    --datasets ai2_arc.arc_challenge hellaswag social_iqa \
    --shots 0 5 \
    --output-dir plots/performance_variations \
    --num-processes 4
```

#### Few-Shot Comparison
```bash
python run_few_shot_comparison.py \
    --models meta-llama/Meta-Llama-3-8B-Instruct \
    --datasets mmlu.college_biology mmlu.high_school_biology \
    --output-dir plots/few_shot_variance
```

#### Prompt Impact Analysis
```bash
python run_prompt_impact_analysis.py \
    --models allenai/OLMoE-1B-7B-0924-Instruct \
    --datasets social_iqa openbook_qa \
    --factors template enumerator separator \
    --shots 0 5
```

#### Robustness Analysis
```bash
python run_robustness_analysis.py \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --datasets mmlu_pro.law mmlu_pro.health \
    --output-dir plots/success_rate_distribution
```

### Programmatic Usage

```python
from plotting import PerformanceAnalyzer, FewShotComparator, PromptImpactAnalyzer, RobustnessAnalyzer
from plotting.utils import DataManager

# Initialize components
with DataManager(use_cache=True) as data_manager:
    analyzer = PerformanceAnalyzer()
    
    # Load data
    data = data_manager.load_multiple_models(
        model_names=['meta-llama/Llama-3.2-1B-Instruct'],
        datasets=['social_iqa'],
        shots_list=[0, 5]
    )
    
    # Create plots
    analyzer.create_performance_plot(
        data=data,
        dataset_name='social_iqa',
        models=['meta-llama/Llama-3.2-1B-Instruct']
    )
```

## Output Structure

```
plots/
├── performance_variations/
│   └── social_iqa/
│       ├── performance_variations.png
│       ├── performance_variations.pdf
│       ├── performance_variations_unified.png
│       └── performance_variations_unified.pdf
├── few_shot_variance/
│   └── social_iqa/
│       ├── zero_few_shot_comparison.png
│       └── zero_few_shot_comparison.svg
├── accuracy_marginalization/
│   └── Llama-3.2-1B-Instruct/
│       └── social_iqa/
│           ├── accuracy_marginalization_0shot.png
│           ├── accuracy_marginalization_5shot.png
│           └── accuracy_marginalization_combined.png
└── success_rate_distribution/
    └── meta_llama_Llama_3_2_1B_Instruct/
        ├── robustness_histogram.png
        └── robustness_histogram.svg
```

## Architecture

The system is organized into four main modules:

### Core Plotters (`plotters/`)
- `performance_analysis.py` - Performance variation analysis
- `few_shot_comparison.py` - Zero-shot vs few-shot comparison  
- `prompt_impact_analysis.py` - Prompt element impact analysis
- `robustness_analysis.py` - Per-question robustness analysis

### Utilities (`utils/`)
- `config.py` - Configuration and constants
- `data_manager.py` - Data loading and caching
- `auth.py` - HuggingFace authentication

### Entry Points
- `run_performance_analysis.py` - Performance analysis runner
- `run_few_shot_comparison.py` - Few-shot comparison runner
- `run_prompt_impact_analysis.py` - Prompt impact analysis runner
- `run_robustness_analysis.py` - Robustness analysis runner

## Configuration

### Default Models
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct  
- meta-llama/Meta-Llama-3-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- allenai/OLMoE-1B-7B-0924-Instruct
- meta-llama/Llama-3.3-70B-Instruct

### Default Datasets
Supports 80+ datasets including:
- **Base**: ai2_arc.arc_challenge, hellaswag, openbook_qa, social_iqa, quality
- **MMLU**: All 57 MMLU subtasks (e.g., mmlu.college_biology, mmlu.high_school_mathematics)
- **MMLU-Pro**: All 14 MMLU-Pro categories (e.g., mmlu_pro.law, mmlu_pro.health)

### Plot Customization
Edit `utils/config.py` to customize:
- Model colors and display names
- Plot styling (fonts, DPI, format)
- Dataset formatting
- Default parameters

## Analysis Types

### 1. Performance Analysis
- **Purpose**: Visualize accuracy distribution across prompt variations
- **Output**: Box plots with scatter points showing performance spread
- **Use Case**: Understanding overall model performance and consistency

### 2. Few-Shot Comparison  
- **Purpose**: Compare 0-shot vs 5-shot performance
- **Output**: Side-by-side box plots with statistical comparison
- **Use Case**: Evaluating few-shot learning effectiveness

### 3. Prompt Impact Analysis
- **Purpose**: Analyze how prompt elements affect performance
- **Elements**: Templates, enumerators, separators, choice ordering
- **Output**: Bar charts showing element-specific impact
- **Use Case**: Optimizing prompt design

### 4. Robustness Analysis
- **Purpose**: Measure per-question consistency across prompts
- **Output**: Histograms showing robustness distribution
- **Use Case**: Identifying reliable vs sensitive questions

## Performance Optimization

### Memory Management
- **Dataset-by-dataset processing**: Prevents memory overflow
- **Smart caching**: Persistent cache directory for faster reruns
- **Parallel processing**: Configurable number of worker processes

### Speed Optimization
```bash
# Use caching for faster subsequent runs  
python run_performance_analysis.py

# Increase parallel processes (if you have sufficient RAM)
python run_performance_analysis.py --num-processes 8

# Process only missing plots
python run_performance_analysis.py  # Automatically skips existing plots
python run_performance_analysis.py --force  # Overwrite existing plots
```

## Troubleshooting

### Common Issues

**Authentication Error**: `invalid_token` or `unauthorized`
```bash
# Verify token is valid
huggingface-cli whoami

# Re-login if needed
huggingface-cli login --token your_new_token
```

**Memory Error**: `MemoryError` or slow performance
```bash
# Reduce parallel processes
python run_analysis.py --num-processes 1

# Process fewer datasets at once
python run_analysis.py --datasets social_iqa  # Single dataset
```

**Missing Data**: `No data found for dataset`
```bash
# Check dataset name spelling
python run_analysis.py --list-datasets

# Verify model availability
python run_analysis.py --list-models
```

**Plot Issues**: Empty or corrupted plots
```bash
# Force regeneration
python run_analysis.py --force

# Check output directory permissions
ls -la plots/
```

### Getting Help

```bash
# List available models
python run_performance_analysis.py --list-models

# List available datasets  
python run_performance_analysis.py --list-datasets

# Get help for specific analysis
python run_performance_analysis.py --help
```

## Development

### Adding New Analysis Types

1. Create new plotter in `plotters/`
2. Add to `__init__.py` imports
3. Create corresponding entry point
4. Update documentation

### Extending Dataset Support

1. Add dataset configuration to `utils/config.py`
2. Update dataset name formatting function
3. Test with new datasets

### Custom Styling

1. Modify `PLOT_STYLE` in `utils/config.py`
2. Update color schemes and fonts
3. Adjust figure sizes and DPI settings

## License

This plotting system is designed for the DOVE dataset evaluation project. Please refer to the main project documentation for licensing information.

## Citation

If you use this plotting system in your research, please cite the DOVE dataset paper and this evaluation framework.

---

For technical support or feature requests, please refer to the main project repository or contact the development team. 