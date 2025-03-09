import itertools
import random
from typing import List, Dict, Any

random.seed(42)


class TemplateVariationDimensions:
    """Configuration parameters for template generation."""

    GREEK_CHARS = "αβγδεζηθικ"
    KEYBOARD_CHARS = "!@#$%^₪*)("

    SHUFFLE_CHOICES_COMBINATIONS = {
        "False": {"shuffle_choices": False, "sort_choices_by_length": False, "sort_choices_alphabetically": False,
                 "reverse_choices": False, "place_correct_choice_position": None},
        "lengthSort": {"shuffle_choices": False, "sort_choices_by_length": True, "sort_choices_alphabetically": False,
                      "reverse_choices": False, "place_correct_choice_position": None},
        "lengthSortReverse": {"shuffle_choices": False, "sort_choices_by_length": True,
                             "sort_choices_alphabetically": False, "reverse_choices": True, "place_correct_choice_position": None},
        "alphabeticalSort": {"shuffle_choices": False, "sort_choices_alphabetically": True,
                           "sort_choices_by_length": False, "reverse_choices": False, "place_correct_choice_position": None},
        "alphabeticalSortReverse": {"shuffle_choices": False, "sort_choices_alphabetically": True,
                                   "sort_choices_by_length": False, "reverse_choices": True, "place_correct_choice_position": None},
        "placeCorrectChoiceFirst": {"shuffle_choices": False, "sort_choices_alphabetically": False,
                                   "sort_choices_by_length": False, "reverse_choices": False, "place_correct_choice_position": 0},
        "placeCorrectChoiceFourth": {"shuffle_choices": False, "sort_choices_alphabetically": False,
                                    "sort_choices_by_length": False, "reverse_choices": False,
                                    "place_correct_choice_position": -1},
    }

    template_dimensions = {
        "enumerator": ["capitals", "lowercase", "numbers", "roman", KEYBOARD_CHARS, GREEK_CHARS],
        "choices_separator": ["\\s", "\n", ", ", "; ", " | ", " OR ", " or "],
        "shuffle_choices": list(SHUFFLE_CHOICES_COMBINATIONS.keys()),
    }

    ENUM_CHARS = {
        "ABCDEFGHIJKLMNOP": "capitals",
        "abcdefghijklmnop": "lowercase",
        str(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20']): "numbers",
        str(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
             'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']): "roman",
        KEYBOARD_CHARS: "keyboard",
        GREEK_CHARS: "greek"
    }

    @classmethod
    def get_options(cls) -> Dict[str, List]:
        """Get all template dimension options.

        Returns:
            Dictionary of dimension names to their possible values
        """
        return cls.template_dimensions

    @classmethod
    def format_value(cls, value: Any) -> str:
        """Format a value for use in template name.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            separator_names = {
                " ": "space",
                "\s": "space",
                "\n": "newline",
                " | ": "pipe",
                " OR ": "OrCapital",
                " or ": "orLower",
                ", ": "comma",
                "; ": "semicolon"
            }

            if value in separator_names:
                return separator_names[value]

            if value in cls.ENUM_CHARS:
                return cls.ENUM_CHARS[value]

            return value.replace(" ", "")

        return str(value)

    @classmethod
    def generate_template_name(cls, combination: Dict[str, Any]) -> str:
        """Generate descriptive template name from parameter combination.

        Args:
            combination: Dictionary of parameter values

        Returns:
            Generated template name
        """
        parts = []
        combination = {key: combination[key] for key in cls.get_options().keys()}
        for key, value in combination.items():
            formatted_value = cls.format_value(value)
            camel_key = cls.to_camel_case(key)
            parts.append(f"{camel_key}_{formatted_value}")
        return "_".join(parts)

    @staticmethod
    def get_shuffle_choices_argument(shuffle_choices_name: str) -> Dict[str, bool]:
        """Get shuffle choices configuration by name.

        Args:
            shuffle_choices_name: Name of shuffle configuration

        Returns:
            Dictionary of shuffle configuration parameters
        """
        return TemplateVariationDimensions.SHUFFLE_CHOICES_COMBINATIONS[shuffle_choices_name]

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase.

        Args:
            snake_str: String in snake_case format

        Returns:
            String in camelCase format
        """
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    @classmethod
    def generate_template_combinations(cls) -> Dict[str, Dict]:
        """Generate all possible combinations of template options.

        Returns:
            Dictionary mapping template names to their parameter combinations
        """
        options = cls.get_options()
        combinations = [
            {key: value for key, value in zip(options.keys(), values)}
            for values in itertools.product(*options.values())
        ]

        return {
            cls.generate_template_name(combo): combo
            for combo in combinations
        }
