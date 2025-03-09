from dataclasses import dataclass

from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicPrompts import BasicPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class RacePrompts(BasicPrompts):
    race_basic: Instruction = Instruction(
        name="MultipleChoiceContextTemplateBasic",
        text=f"Context: {{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:"
    )

    race_basic_no_context_label: Instruction = Instruction(
        name="MultipleChoiceContextTemplateBasicNoContextLabel",
        text=f"{{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:"
    )

    race_mmlu_style: Instruction = Instruction(
        name="MultipleChoiceContextTemplateMMluStyle",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    race_mmlu_helm_style: Instruction = Instruction(
        name="MultipleChoiceContextTemplateMMluHelmStyle",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )

    race_mmlu_helm_with_choices: Instruction = Instruction(
        name="MultipleChoiceContextTemplateMMluHelmWithChoices",
        text=f"The following are multiple choice questions (with answers).\n\nContext: {{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )

    race_prosa_simple: Instruction = Instruction(
        name="MultipleChoiceContextTemplateProSASimple",
        text=f"Please answer the following question based on the article:\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    race_prosa_could_you: Instruction = Instruction(
        name="MultipleChoiceContextTemplateProSACould",
        text=f"Could you provide a response to the following question based on the article:\nContext: {{context}}\n{{question}}\n{{choices}}\nAnswer:"
    )

    race_state_numbered: Instruction = Instruction(
        name="MultipleChoiceContextTemplateStateNumbered",
        text=(
            f"Answer the multiple choice Question from one of the Choices (choose from numerals) based on the context.\n"
            f"Context: {{context}}\nQuestion: {{question}}\nChoices:\n{{choices}}\nAnswer:")
    )

    race_state_options: Instruction = Instruction(
        name="MultipleChoiceContextTemplateStateOptions",
        text=f"Based on the provided article, please respond to the following question:\nContext: {{context}}\nQuestion: {{question}}\nOptions: {{choices}}\nAnswer:"
    )

    race_state_select: Instruction = Instruction(
        name="MultipleChoiceContextTemplateStateSelect",
        text=f"Select the correct answer for the following multiple-choice question based on the context. Choose from the numbered options provided.\nContext: {{context}}\nQuestion: {{question}}\nOptions: {{choices}}\nAnswer:"""
    )

    race_state_read: Instruction = Instruction(
        name="MultipleChoiceContextTemplateStateRead",
        text=f"Read the following context and answer the question:\n{{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )

    race_state_multiple_choice: Instruction = Instruction(
        name="MultipleChoiceContextTemplateStateMultipleChoice",
        text=f"Answer the multiple-choice question below about the context:\n\nContext: {{context}}\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )
