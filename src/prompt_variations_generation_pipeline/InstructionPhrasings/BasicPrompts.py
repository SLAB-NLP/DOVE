from dataclasses import dataclass, fields

from src.prompt_variations_generation_pipeline.InstructionPhrasings.Instruction import Instruction


@dataclass(frozen=True)
class BasicPrompts:
    def get_instruction_phrasings(self):
        return [
            getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Instruction)
        ]
