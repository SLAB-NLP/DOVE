from dataclasses import dataclass

from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicPrompts import BasicPrompts
from src.prompt_variations_generation_pipeline.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class HellaSwagPrompts(BasicPrompts):
    hellaswag_instructions_standard: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStandard",
        text=f"Pick the best ending to the sentence.\nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"
    )

    hellaswag_instructions_context: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsContext",
        text=f"Pick the best ending to the context.\nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"
    )

    hellaswag_instructions_structured: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsStructured",
        text=f"Context: [context] Choices: [choices] Answer: [answer]\nContext: {{context}} Choices:\n{{choices}}\nAnswer:"
    )

    hellaswag_instructions_basic: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsBasic",
        text=f"Context: {{context}}\n\nChoices: {{choices}}\nAnswer:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_1: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState1",
        text=f"Complete the following scenario by selecting the most appropriate ending.\n\nContext: {{context}}\n\nChoices:\n{{choices}}\nAnswer:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_2: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState2",
        text=f"Select the most suitable conclusion for the sentence given the context: {{context}}. Here are your options: {{choices}}. Please provide your answer."
    )

    hellaswag_paraphrase_State_of_What_Art_paper_3: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState3",
        text=f"Choose the most suitable ending for the sentence based on the given context.\nContext: {{context}}\nOptions:\n{{choices}}\nYour Answer:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_4: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState4",
        text=f"Select the most suitable conclusion for the sentence.\nGiven Context: {{context}}\nOptions:\n{{choices}}\nResponse:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_5: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState5",
        text=f"Choose the most suitable conclusion for the sentence.\nContext: {{context}}\nOptions:\n{{choices}}\nResponse:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_6: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState6",
        text=f"Given the context and choices provided, select the most appropriate ending for the sentence. Use the context to understand the situation and choose the option that best completes the sentence. Context: {{context}} Choices: {{choices}} Answer:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_7: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState7",
        text=f"Pick the best ending to the sentence based on the given context and choices. Use the context provided to determine the most suitable choice. \nContext: {{context}}\nChoices:\n{{choices}}\nAnswer:"
    )

    hellaswag_paraphrase_State_of_What_Art_paper_8: Instruction = Instruction(
        name="MultipleChoiceTemplatesInstructionsState8",
        text=f"Based on the provided context, choose the most suitable ending for the sentence from the given options. Context: {{context}} Choices: {{choices}} Answer:"
    )
