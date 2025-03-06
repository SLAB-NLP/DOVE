# 🕊️ DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation

DOVE is a large-scale dataset containing prompt perturbations of various evaluation benchmarks designed to study LLM sensitivity from a holistic perspective.

[![Paper](https://img.shields.io/badge/arxiv-paper-red)](https://arxiv.org/abs/2503.01622)
[![Dataset](https://img.shields.io/badge/🤗-dataset-yellow)](https://huggingface.co/datasets/nlphuji/DOVE)
[![Website](https://img.shields.io/badge/🌐-website-blue)](https://slab-nlp.github.io/DOVE/)
[![Contact](https://img.shields.io/badge/📧-contact-green)](mailto:eliyahaba@mail.huji.ac.il)

## Abstract

Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices.

In this work, we present DOVE (Dataset Of Variation Evaluation), a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from a holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations.

DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation.

## Dataset Access

Browse the data: [https://huggingface.co/datasets/nlphuji/DOVE](https://huggingface.co/datasets/nlphuji/DOVE)

## Usage

```python
from datasets import load_dataset

# Load a specific model/language/shots benchmark
def load_benchmark(repo_id, model_name, language="en", shots=0, benchmark_file="mmlu.global_facts.parquet"):
    file_path = f"{model_name}/{language}/{shots}_shot/{benchmark_file}"
    return load_dataset(repo_id, data_files=file_path, split="train")

# Examples
# Example 1: Loading from Dove_Lite repository
llama_en_arc_challenge = load_benchmark("nlphuji/DOVE_Lite", "Meta-Llama-3-8B-Instruct", "en", 0, "ai2_arc.arc_challenge.parquet")

# Example 2: Loading from full Dove repository
mistral_ja_formal_logic = load_benchmark("nlphuji/DOVE", "Mistral-7B-Instruct-v0.3", "en", 5, "mmlu.formal_logic.parquet")

# Print dataset information
print(f"Dataset loaded successfully:")
print(f"- Llama (en) arc_challenge: {len(llama_en_arc_challenge)} examples")
print(f"- Mistral (en) formal_logic: {len(mistral_en_formal_logic)} examples")
```

## Dataset Structure

```
nlphuji/
├── Dove/
│   ├── model_name/                      # e.g., "Llama-3.2-1B-Instruct"
│   │   ├── language/                    # e.g., "en", "fr"
│   │   │   └── shots_N/                 # N = 0 for zero-shot, N > 0 for few-shot
│   │   │       ├── mmlu.abstract_algebra.parquet
│   │   │       ├── mmlu.world_religions.parquet
│   │   │       ├── ai2_arc.arc_challenge.parquet
│   │   │       ├── hellaswag.parquet
│   │   │       └── other_benchmark_files.parquet
│   └── other_models/
└── Dove_Lite/
    └── [same structure with reduced metadata per instance]
```

## Versions

### Full Version (2TB)
- Complete token-level probabilities
- Detailed few-shot examples
- Comprehensive model behavior analysis

[Download Full Version](https://huggingface.co/datasets/nlphuji/DOVE)

### Lite Version (100GB)
- Core prompt variations
- Model responses and Evaluation scores
- Perfect for quick experimentation

[Download Lite Version](https://huggingface.co/datasets/nlphuji/DOVE_Lite)

## Contribute to DOVE

Help develop meaningful protocols for systematic LLM evaluation.

### Why Contribute?
- Democratize the systematic study of LLM sensitivity
- Spur research into meaningful evaluation
- Enable efficient and generalizable evaluation
- Join our growing, community-driven resource

### How to Contribute?
- Share model predictions across any prompt variation
- Contribute data from diverse domains and tasks
- Help expand coverage across languages
- Explore additional evaluation dimensions
- Suggest directions for future research

Contributors who provide significant data will be invited as co-authors on the next version of both the paper and dataset.

Your data doesn't need to include every field in our [Dataset Scheme](https://arxiv.org/pdf/XXXX.XXXXX.pdf#page=20), but should demonstrate systematic evaluation by including model predictions with variations in at least one dimension.

To share your data, contact: [eliyahaba@mail.huji.ac.il](mailto:eliyahaba@mail.huji.ac.il)

## Citation

```bibtex
@misc{habba2025dovelargescalemultidimensionalpredictions,
      title={DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation}, 
      author={Eliya Habba and Ofir Arviv and Itay Itzhak and Yotam Perlitz and Elron Bandel and Leshem Choshen and Michal Shmueli-Scheuer and Gabriel Stanovsky},
      year={2025},
      eprint={2503.01622},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01622}, 
}
```

## License

This dataset is licensed under the **Computational Data License Agreement v2 (CDLAv2)**. For full license terms, see: [https://cdla.dev/permissive-2.0/](https://cdla.dev/permissive-2.0/)
