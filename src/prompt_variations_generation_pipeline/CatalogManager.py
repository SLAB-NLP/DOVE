import argparse
from pathlib import Path

from unitxt import add_to_catalog, get_from_catalog
from unitxt.templates import Template, MultipleChoiceTemplate

from src.prompt_variations_generation_pipeline.configuration_generation.Constants import Constants


class CatalogManager:
    """Manages saving and loading templates to/from the Unitxt local catalog."""

    def __init__(self, catalog_path: Path) -> None:
        """Initialize catalog manager.

        Args:
            catalog_path: Path to the local catalog directory
        """
        self.catalog_path = catalog_path

    def save_to_catalog(self, template: Template, name: str) -> None:
        """Save template to local catalog.

        Args:
            template: Template instance to save
            name: Name to save the template under
        """
        add_to_catalog(template, name, catalog_path=str(self.catalog_path), overwrite=True)
        print(f"Dataset saved successfully to local catalog: {self.catalog_path}")

    def load_from_catalog(self, name: str) -> Template:
        """Load template from local catalog.

        Args:
            name: Name of template to load

        Returns:
            Template instance from catalog
        """
        return get_from_catalog(name, catalog_path=str(self.catalog_path))


if __name__ == "__main__":
    # Define the path to your local catalog directory within your repository (replace with your actual path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_path", type=str, default=Constants.CATALOG_PATH)
    parser.add_argument("--template_name", type=str, default="my_custom_qa_template")
    args = parser.parse_args()

    # Create an instance of the DatasetSaver
    catalog_manager = CatalogManager(args.catalog_path)

    # Define the MultipleChoiceTemplate (assuming your data adheres to this format)

    template = MultipleChoiceTemplate(
        input_format="The following are multiple choice questions (with answers) about {topic}"
                     ".\n\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:",
        target_field="answer",
        choices_separator="\n",
        enumerator="numerals",
        postprocessors=["processors.first_character"],
    )

    # Save the dataset using the saver
    catalog_manager.save_to_catalog(template, args.template_name)
    my_task = get_from_catalog(args.template_name, catalog_path=args.catalog_path)
