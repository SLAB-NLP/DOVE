import argparse
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

from src.prompt_variations_generation_pipeline.CatalogManager import CatalogManager
from src.prompt_variations_generation_pipeline.configuration_generation.Constants import Constants
from src.prompt_variations_generation_pipeline.configuration_generation.DatasetConfigFactory import DatasetConfigFactory
from src.prompt_variations_generation_pipeline.configuration_generation.MultipleChoiceTemplateGenerator import MultipleChoiceTemplateGenerator
from src.prompt_variations_generation_pipeline.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate multiple choice templates')
    parser.add_argument('--catalog_path', type=Path, default=Constants.CATALOG_PATH,
                       help='Path to save the generated templates')
    parser.add_argument('--datasets', nargs='+', default=['MMLU', 'HellaSwag', 'RACE'],
                       help='List of datasets to generate templates for')
    args = parser.parse_args()

    # Get configurations for all specified datasets
    dataset_configs = {
        name: DatasetConfigFactory.get_instruct_prompts(name)
        for name in args.datasets
    }

    # Template variation options
    override_options = TemplateVariationDimensions.template_dimensions

    # Generate templates for each dataset
    for dataset_name, dataset_config in dataset_configs.items():
        print(colored(f"\nGenerating templates for {dataset_name}...", "blue"))
        
        # Get instruction phrasings for this dataset
        instruction_phrasings = dataset_config.get_instruction_phrasings()
        
        # Initialize catalog manager
        catalog_manager = CatalogManager(args.catalog_path)
        
        # Generate templates for each instruction phrasing
        for instruction in instruction_phrasings:
            try:
                # Initialize template generator
                generator = MultipleChoiceTemplateGenerator(
                    dataset_config,
                    override_options,
                    instruction.text
                )
                
                # Generate all template variations
                templates = generator.create_templates()
                
                # Save templates to catalog
                for template_name, template in tqdm(templates.items(), 
                                                 desc=f"Saving {instruction.name} templates"):
                    catalog_manager.save_to_catalog(
                        template,
                        f"{dataset_name}.{instruction.name}.{template_name}"
                    )
                
                # Generate metadata
                generator.create_and_process_metadata(
                    list(templates.values()),
                    dataset_name,
                    override_options
                )
                
                print(colored(f"Successfully generated templates for {dataset_name} - {instruction.name}", "green"))
                
            except Exception as e:
                print(colored(
                    f"Error generating templates for {dataset_name} - {instruction.name}: {str(e)}",
                    "red"
                ))
                continue


if __name__ == "__main__":
    main() 