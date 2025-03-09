from dataclasses import dataclass

from src.prompt_variations_generation_pipeline.InstructionPhrasings.BasicMCPrompts import BasicMCPrompts


@dataclass(frozen=True)
class MMLUPrompts(BasicMCPrompts):
    pass
