import argparse
from copy import deepcopy
from dataclasses import asdict
from typing import List

from termcolor import colored
from tqdm import tqdm
from unitxt.templates import MultipleChoiceTemplate
from unitxt.templates import Template

from src.prompt_variations_generation_pipeline.CatalogManager import CatalogManager
from src.prompt_variations_generation_pipeline.configuration_generation.Constants import Constants
from src.prompt_variations_generation_pipeline.configuration_generation.DatasetConfigFactory import DatasetConfigFactory
from src.prompt_variations_generation_pipeline.configuration_generation.InputTemplatesConfigs.MultipleChoiceTemplateConfig import \
    MultipleChoiceTemplateConfigFactory
from src.prompt_variations_generation_pipeline.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions
from src.prompt_variations_generation_pipeline.configuration_generation.TemplateGenerator import TemplateGenerator



class MultipleChoiceTemplateGenerator(TemplateGenerator):
    def create_template(self, **override_args) -> MultipleChoiceTemplate:
        """Creates a MultipleChoiceTemplate instance with the given parameters.

        Args:
            **override_args: Override options for template parameters.

        Returns:
            MultipleChoiceTemplate: The configured template instance.
        """
        input_config = MultipleChoiceTemplateConfigFactory.create({
            **override_args,
        })
        template = MultipleChoiceTemplate(**asdict(input_config))
        return template

    def create_and_process_metadata(self, created_templates: List[Template], dataset_name: str,
                                    override_options: dict) -> None:
        """Creates and processes metadata for the given templates.
        Processes the metadata by escaping special characters in separators
        and converting enumerator values to their string representations.

        Args:
            created_templates: List of template instances
            dataset_name: Name of the dataset
            override_options: Dictionary of template parameter options

        Saves the processed metadata to a CSV file.
        """
        metadata_df = self.create_metadata_from_templates(created_templates, params=override_options)
        
        metadata_df['choices_separator'] = metadata_df['choices_separator'].replace(' ', '\\s')
        metadata_df['choices_separator'] = metadata_df['choices_separator'].replace('\n', '\\n')
        
        metadata_df['enumerator'] = metadata_df['enumerator'].astype(str)
        metadata_df.replace({"enumerator": TemplateVariationDimensions.ENUM_CHARS}, inplace=True)
        


if __name__ == "__main__":
    catalog_path = Constants.CATALOG_PATH
    parser = argparse.ArgumentParser(description='Generate multiple choice templates')
    dataset_names_to_configs = DatasetConfigFactory.get_all_instruct_prompts()

    for dataset_name, datasetConfig in dataset_names_to_configs.items():
        override_options = deepcopy(TemplateVariationDimensions.template_dimensions)
        instruction_phrasing_data = datasetConfig.get_instruction_phrasings()
        
        for instruction_phrasing in instruction_phrasing_data:
            instruct_folder_name = instruction_phrasing.name
            input_format = instruction_phrasing.text
            try:
                print(colored(f"Creating templates for {dataset_name}", "blue"))
                generator = MultipleChoiceTemplateGenerator(datasetConfig, override_options, input_format)
                created_templates = generator.create_templates()

                catalog_manager = CatalogManager(catalog_path)
                for template_name, template in tqdm(created_templates.items()):
                    catalog_manager.save_to_catalog(template,
                                                 f"{dataset_name}.{instruct_folder_name}.{template_name}")

                generator.create_and_process_metadata(list(created_templates.values()), dataset_name, override_options)
                print(colored(f"Templates for {dataset_name} created successfully", "green"))
            except Exception as e:
                print(colored(
                    f"Error in creating templates for {dataset_name} with function {input_format}: {e}",
                    "red"))
                continue
