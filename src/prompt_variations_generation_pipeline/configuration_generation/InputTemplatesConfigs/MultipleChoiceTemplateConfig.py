from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.prompt_variations_generation_pipeline.configuration_generation.InputTemplatesConfigs.InputFormatTemplateConfig import \
    InputFormatTemplateConfig
from src.prompt_variations_generation_pipeline.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions


@dataclass
class MultipleChoiceTemplateConfig(InputFormatTemplateConfig):
    input_format: str
    choices_field: str = "choices"
    target_field: str = "answer"
    choices_separator: str = "\n"
    enumerator: str = "numbers"
    source_choice_format: str = "{choice_numeral}. {choice_text}"
    target_choice_format: str = "{choice_numeral}. {choice_text}"
    postprocessors: List[str] = field(
        default_factory=lambda: [
            "processors.to_string_stripped",
            "processors.take_first_non_empty_line",
            "processors.match_closest_option"
        ]
    )
    shuffle_choices: bool = False
    shuffle_choices_seed: Optional[int] = None
    sort_choices_by_length: bool = False
    sort_choices_alphabetically: bool = False
    reverse_choices: bool = False
    place_correct_choice_position: Optional[int] = None


class MultipleChoiceTemplateConfigFactory:
    @staticmethod
    def create(kwargs: Dict[str, Any]) -> MultipleChoiceTemplateConfig:
        """Create a MultipleChoiceTemplateConfig instance from kwargs."""
        extracted_kwargs = kwargs.copy()

        # Process shuffle choices
        shuffle_choices_str = extracted_kwargs.pop("shuffle_choices")
        shuffle_choices_config = MultipleChoiceTemplateConfigFactory._get_shuffle_choices_config(shuffle_choices_str)

        # Merge all kwargs
        config_kwargs = {
            **extracted_kwargs,
            **shuffle_choices_config
        }

        return MultipleChoiceTemplateConfig(**config_kwargs)

    @staticmethod
    def _get_shuffle_choices_config(shuffle_choices: str) -> Dict[str, Any]:
        """Get shuffle choices configuration."""
        return TemplateVariationDimensions.get_shuffle_choices_argument(shuffle_choices)
