# Multiple Choice Template Generator

A module for generating systematic variations of multiple-choice question templates, used in the DOVE dataset creation pipeline.

## Overview

The `MultipleChoiceTemplateGenerator` creates templates with controlled variations along multiple dimensions:
- Instruction phrasing (different ways to phrase the task)
- Choice separators (newline, comma, etc.)
- Enumerators (letters, numbers, etc.) 
- Choice ordering methods

## Components

### MultipleChoiceTemplateGenerator

Main class for template generation:

```python
def create_template(**override_args) -> MultipleChoiceTemplate:
    """Creates a single template with specified parameters"""

def create_templates() -> Dict[str, Template]:
    """Creates all template variations based on configuration options"""

def create_and_process_metadata(templates, dataset_name, options) -> None:
    """Processes and saves template metadata to CSV"""
```

### Template Variation Dimensions

Defines the template variation dimensions in `TemplateVariationDimensions`:

1. **Instruction Phrasing**:
   - Dataset-specific instruction variations (e.g., MMLU, HellaSwag, RACE)
   - Different ways to phrase the multiple-choice task
   - Controlled through dataset-specific prompt classes

2. **Enumerators**:
   - `capitals`: A, B, C, ...
   - `lowercase`: a, b, c, ...
   - `numbers`: 1, 2, 3, ...
   - `roman`: I, II, III, ...
   - `greek`: α, β, γ, ...
   - `keyboard`: !, @, #, ...

3. **Choice Separators**:
   - Newline (`\n`)
   - Comma (`, `)
   - Semicolon (`; `)
   - Pipe (` | `)
   - OR variants (` OR `, ` or `)
   - Space (`\s`)

4. **Choice Ordering**:
   - Original order (`False`)
   - Length-based sorting (`lengthSort`, `lengthSortReverse`)
   - Alphabetical sorting (`alphabeticalSort`, `alphabeticalSortReverse`)
   - Position-based (`placeCorrectChoiceFirst`, `placeCorrectChoiceFourth`)

## Usage

```python
from src.prompt_variations_generation_pipeline.configuration_generation.MultipleChoiceTemplateGenerator import MultipleChoiceTemplateGenerator
from src.prompt_variations_generation_pipeline.configuration_generation.DatasetConfigFactory import DatasetConfigFactory
from src.prompt_variations_generation_pipeline.CatalogManager import CatalogManager

# Get dataset configuration with instruction phrasings
dataset_config = DatasetConfigFactory.get_instruct_prompts("MMLU")
instruction_phrasings = dataset_config.get_instruction_phrasings()

# For each instruction phrasing
for instruction in instruction_phrasings:
    # Initialize generator
    generator = MultipleChoiceTemplateGenerator(
        dataset_config,
        override_options,
        instruction.text
    )

    # Generate all template variations
    templates = generator.create_templates()

    # Save to catalog
    catalog_manager = CatalogManager(catalog_path)
    for name, template in templates.items():
        catalog_manager.save_to_catalog(
            template, 
            f"{dataset_name}.{instruction.name}.{name}"
        )
```

## Template Naming

Templates are named based on their configuration parameters:
```
{dataset}.{instruction_name}.enumerator_{type}_choicesSeparator_{type}_shuffleChoices_{type}
```

Example: `MMLU.formal.enumerator_numbers_choicesSeparator_newline_shuffleChoices_lengthSort`

## Metadata

Generates a CSV file with template metadata including:
- Instruction phrasing variations
- Parameter configurations
- Formatting options
- Choice ordering methods

## Integration

This module is part of the DOVE dataset creation pipeline, used to generate systematic prompt variations for evaluating LLM robustness across different formatting dimensions. 