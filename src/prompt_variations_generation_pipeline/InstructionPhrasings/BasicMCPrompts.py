from dataclasses import dataclass

from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicPrompts import BasicPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class BasicMCPrompts(BasicPrompts):
    instructions_with_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopic",
        text=f"The following are multiple choice questions (with answers) about {{topic}}.\n\n{{question}}\n{{choices}}\nAnswer:"
    )

    instructions_without_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithoutTopic",
        text=f"The following are multiple choice questions (with answers).\n\n{{question}}\n\n{{choices}}\nAnswer:"
    )

    instructions_without_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithoutTopicFixed",
        text=f"The following are multiple choice questions (with answers).\n\n{{question}}\n{{choices}}\nAnswer:"
    )

    instructions_with_topic_helm: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopicHelm",
        text=f"The following are multiple choice questions (with answers) about {{topic}}.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )

    instructions_without_topic_helm: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithoutTopicHelm",
        text=f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n\n{{choices}}\nAnswer:"
    )

    instructions_without_topic_helm: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithoutTopicHelmFixed",
        text=f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )

    instructions_without_topic_lm_evaluation_harness: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithoutTopicHarness",
        text=f"Question: {{question}}\n\nChoices: {{choices}}\nAnswer:"
    )

    structured_instruction_with_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesStructuredWithTopic",
        text=f"Topic: {{topic}}\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
    )

    structured_instruction_without_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesStructuredWithoutTopic",
        text=f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
    )

    # mmlu_instructions_with_topic_and_cot: Instruction = Instruction(
    #     name="MultipleChoiceTemplatesInstructionsWithTopicAndCoT",
    #     text=(f"The following are multiple choice questions (with answers) about {{topic}}. Think step by"
    #           f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
    # )

    please_simple_prompt_ProSA_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsProSASimple",
        text=f"Please answer the following question:\n{{question}}\n{{choices}}\nAnswer:"
    )

    please_address_prompt_ProSA_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsProSAAddress",
        text=f"Please address the following question:\n{{question}}\n{{choices}}\nAnswer:"
    )

    could_you_prompt_ProSA_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsProSACould",
        text=f"Could you provide a response to the following question:\n{{question}}\n{{choices}}\nAnswer:"
    )

    here_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateHere",
        text=f"Here are some multiple choice questions along with their answers about {{topic}}.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
    )

    below_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateBelow",
        text=f"Below are multiple-choice questions related to {{topic}}, each followed by their respective answers.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
    )

    below_please_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateBelowPlease",
        text=f"Below are multiple-choice questions related to {{topic}}. Please provide the correct answer for each question.\n\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )
