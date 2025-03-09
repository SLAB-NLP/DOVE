class Instruction:
    def __init__(self, name: str, text: str):
        self.name = name
        self.text = text

    def __repr__(self):
        return f"PromptInstruction(name={self.name}, text={self.text})"
