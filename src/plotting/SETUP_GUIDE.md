# 🚀 Quick Setup Guide

A streamlined guide to get the DOVE Plotting System running in under 5 minutes.

## ⚡ Fast Installation

### 1. Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv dove-plotting
source dove-plotting/bin/activate  # Linux/Mac
# dove-plotting\Scripts\activate    # Windows

# Navigate to plotting directory
cd src/analysis/plotting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. HuggingFace Authentication
Choose one option:

**Option A: Environment Variable**
```bash
export HF_ACCESS_TOKEN="hf_your_token_here"
```

**Option B: HuggingFace CLI**
```bash
pip install huggingface_hub[cli]
huggingface-cli login
```

**Option C: .env File**
```bash
# Create .env file in the plotting directory
echo 'HF_ACCESS_TOKEN="hf_your_token_here"' > .env
```

### 4. Optional: Set Cache Directory
```bash
# Set custom cache directory (optional)
export DOVE_CACHE_DIR="/path/to/your/cache"
```

## 🎯 Quick Test

Test the system with minimal data:

```bash
# Quick test with 1 model, 1 dataset, minimal shots
python run_performance_analysis.py \
  --models meta-llama/Llama-3.2-1B-Instruct \
  --datasets social_iqa \
  --shots 0 \
  --num-processes 1
```

## 📁 Verify Installation

Check that these work without errors:

```bash
# List available models
python run_performance_analysis.py --list-models

# List available datasets  
python run_performance_analysis.py --list-datasets

# Check help
python run_performance_analysis.py --help
```

## 🎨 Generate Your First Plots

Run with conservative settings:

```bash
# Performance variations (safest to start with)
python run_performance_analysis.py \
  --models meta-llama/Llama-3.2-1B-Instruct \
  --datasets social_iqa \
  --output-dir test_plots

# Check output
ls test_plots/social_iqa/
```

You should see:
```
test_plots/social_iqa/
├── performance_variations.png
├── performance_variations.pdf
├── performance_variations_unified.png
└── performance_variations_unified.pdf
```

## 🎉 Test All Analysis Types

```bash
# Performance analysis
python run_performance_analysis.py --datasets social_iqa

# Few-shot comparison  
python run_few_shot_comparison.py --datasets social_iqa

# Prompt impact analysis
python run_prompt_impact_analysis.py --datasets social_iqa

# Robustness analysis
python run_robustness_analysis.py --datasets social_iqa
```

## 🔧 Common Setup Issues

### Missing HuggingFace Token
**Error**: `Repository not found` or `Authentication required`
**Solution**: Ensure your HuggingFace token has access to the DOVE dataset

### Memory Issues
**Error**: `OOM` or system freezing
**Solution**: 
```bash
# Use single process and minimal datasets
python run_performance_analysis.py --num-processes 1 --datasets social_iqa
```

### Import Errors
**Error**: `ModuleNotFoundError`
**Solution**: 
```bash
# Ensure you're in the right directory
cd src/analysis/plotting

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Permission Errors
**Error**: Cannot write to output directory
**Solution**:
```bash
# Use custom output directory
python run_performance_analysis.py --output-dir ~/my_plots
```

### Cache Directory Issues
**Error**: Permission denied when creating cache
**Solution**:
```bash
# Set custom cache directory
export DOVE_CACHE_DIR="~/dove_cache"
mkdir -p ~/dove_cache
```

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list | grep matplotlib`)
- [ ] HuggingFace authentication working (`huggingface-cli whoami`)
- [ ] Test script runs without errors
- [ ] Sample plots generated successfully
- [ ] All 4 plotting scripts are executable

## 🎉 Next Steps

Once setup is complete:

1. **Explore the full README.md** for detailed usage
2. **Try different analysis types** with your data
3. **Customize configurations** in `utils/config.py`  
4. **Scale up** to full model/dataset combinations

## 📊 Available Analysis Types

1. **Performance Analysis** (`run_performance_analysis.py`)
   - Box plots showing accuracy distribution
   - Two versions: combined and unified plots

2. **Few-Shot Comparison** (`run_few_shot_comparison.py`)
   - Side-by-side 0-shot vs 5-shot comparison
   - Statistical analysis of few-shot improvements

3. **Prompt Impact Analysis** (`run_prompt_impact_analysis.py`)
   - Effect of different prompt elements
   - Template, enumerator, separator analysis

4. **Robustness Analysis** (`run_robustness_analysis.py`)
   - Per-question consistency analysis
   - Success rate distributions

**🎯 You're ready to start plotting!** 📊 