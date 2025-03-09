from src.prompt_variations_generation_pipeline.InstructionPhrasings.AI2ARCPrompts import AI2ARCPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.HellaSwagPrompts import \
    HellaSwagPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.MMLUPrompts import MMLUPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.OpenBookQAPrompts import \
    OpenBookQAPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.QuALITYPrompts import QuALITYPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.RacePrompts import RacePrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.SocialQaPrompts import \
    SocialQaPrompts


class DatasetConfigFactory:
    @staticmethod
    def get_all_instruct_prompts():
        """Get instruction prompts for all supported datasets.

        Returns:
            Dictionary mapping dataset names to their prompt configurations
        """
        dataset_instruct_prompts = {
            'AI2_ARC': AI2ARCPrompts(),
            'HellaSwag': HellaSwagPrompts(),
            'MMLU': MMLUPrompts(),
            'MMLU_PRO': MMLUPrompts(),
            'OpenBookQA': OpenBookQAPrompts(),
            'Social_IQa': SocialQaPrompts(),
            'Race': RacePrompts(),
            "QuALITY": QuALITYPrompts(),
            # Add other datasets here
        }
        return dataset_instruct_prompts

    @staticmethod
    def get_instruct_prompts(dataset_name):
        """Get instruction prompts for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Prompt configuration for the dataset, or BasicMCPrompts if not found
        """
        dataset_instruct_prompts = {
            'AI2_ARC': AI2ARCPrompts(),
            'HellaSwag': HellaSwagPrompts(),
            'MMLU': MMLUPrompts(),
            'MMLU_PRO': MMLUPrompts(),
            'OpenBookQA': OpenBookQAPrompts(),
            'Social_IQa': SocialQaPrompts(),
            'Race': RacePrompts(),
            'QuALITY': QuALITYPrompts(),
            # Add other datasets here
        }
        return BasicMCPrompts() if dataset_name not in dataset_instruct_prompts else \
            dataset_instruct_prompts[dataset_name]
