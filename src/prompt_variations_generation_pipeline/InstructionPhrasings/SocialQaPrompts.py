from dataclasses import dataclass

from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class SocialQaPrompts(BasicMCPrompts):
    instructions_with_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopic",
        text=f"The following are multiple choice questions (with answers) about common sense social interactions.\n\n{{question}}\n{{choices}}\nAnswer:"
    )

    instructions_with_topic_helm: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopicHelm",
        text=f"The following are multiple choice questions (with answers) about common sense social interactions.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
    )

    structured_instruction_with_topic: Instruction = Instruction(
        name="MultipleChoiceTemplatesStructuredWithTopic",
        text=f"Topic: common sense social interactions\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
    )

    mmlu_instructions_with_topic_and_cot: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsWithTopicAndCoT",
        text=(f"The following are multiple choice questions (with answers) about common sense social interactions. Think step by"
              f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
    )

    here_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateHere",
        text=f"Here are some multiple choice questions along with their answers about common sense social interactions.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
    )

    below_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateBelow",
        text=f"Below are multiple-choice questions related to common sense social interactions, each followed by their respective answers.\n\nQuestion: {{question}}\nChoices: {{choices}}\nCorrect Answer:"
    )

    below_please_prompt_State_of_What_Art_paper: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStateBelowPlease",
        text=f"Below are multiple-choice questions related to common sense social interactions. Please provide the correct answer for each question.\n\nQuestion: {{question}}\nChoices: {{choices}}\nAnswer:"
    )
