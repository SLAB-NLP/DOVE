# DOVE Graph Images

This directory contains the visualization graphs for the DOVE website.

## Required Files

Place your graph images here with the following exact filenames:

### Tab 1: Model Sensitivity
- `sensitivity-analysis.png` - Main sensitivity analysis chart

### Tab 2: Model Comparison  
- `model-comparison.png` - Model performance comparison chart

### Tab 3: Prompt Effects
- `prompt-dimensions.png` - Prompt dimension effects visualization
- `few-shot-analysis.png` - Few-shot learning impact chart

### Tab 4: Benchmark Analysis
- `benchmark-analysis.png` - Performance across benchmarks

### Tab 5: Key Insights
- `key-insights.png` - Research insights overview chart

## Image Specifications

**Recommended specifications:**
- **Format:** PNG (supports transparency)
- **Width:** 800-1200px (will be responsive)
- **Height:** 400-600px 
- **Resolution:** 72-150 DPI (web optimized)
- **Background:** White or transparent

## How to Add Your Graphs

1. **Export your graphs** from Python/R/MATLAB with the filenames above
2. **Copy them** to this directory
3. **Refresh the website** - they will appear automatically in the tabs

## Example Python Code to Save Graphs

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create your plot
plt.figure(figsize=(12, 8))
# ... your plotting code ...

# Save with correct filename
plt.savefig('static/images/graphs/sensitivity-analysis.png', 
           dpi=150, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()
```

## Auto-Update Feature

The website automatically shows the last modification date of the graphs, so when you update any graph file, visitors will see the updated timestamp.

## Fallback

If any image is missing, the website will show a placeholder indicating where the graph should appear. 