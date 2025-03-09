from abc import abstractmethod
from itertools import product
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from unitxt.templates import Template

from src.prompt_variations_generation_pipeline.configuration_generation.InputTemplatesConfigs.InputFormatTemplateConfig import \
    InputFormatTemplateConfig
from src.prompt_variations_generation_pipeline.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions


class TemplateGenerator:
    def __init__(self, input_config: InputFormatTemplateConfig, override_options: dict, input_format: str):
        """Initialize the template generator.

        Args:
            input_config: Configuration for input format template
            override_options: Dictionary of parameter options to override
            input_format: Format string for the input template
        """
        self.input_config = input_config
        self.override_options = override_options
        self.input_format = input_format

    def create_templates(self) -> Dict[str, Template]:
        """Create templates with all combinations of override parameters.

        Returns:
            Dict mapping template names to Template instances
        """
        templates = {}
        for options in tqdm(product(*self.override_options.values())):
            override_args = dict(zip(self.override_options.keys(), options))
            # create a name for the template
            template_name = TemplateVariationDimensions.generate_template_name(override_args)
            override_args['input_format'] = self.input_format
            template = self.create_template(**override_args)
            templates[template_name] = template
            print(f"Created template with options: {override_args}")
        print("All templates created!")
        return templates

    @staticmethod
    def create_metadata_from_templates(templates: List[Template], params: dict) -> pd.DataFrame:
        """Create metadata DataFrame from templates.

        Args:
            templates: List of template instances
            params: Dictionary of template parameters
        
        Returns:
            DataFrame containing template metadata
        """
        # create a df that contains the templates and their parameter

        columns = list(params.keys())
        # del columns[columns.index('shuffle_choices')]
        rows = []
        for i, template in tqdm(enumerate(templates)):
            row = []
            row.append(f"template_{i}")
            template_params = [getattr(template, column) for column in columns]
            row.extend(template_params)
            rows.append(row)
        df = pd.DataFrame(rows, columns=['template_name'] + columns)
        df.set_index('template_name', inplace=True)
        return df

    @abstractmethod
    def create_template(self, **override_args) -> Template:
        """
        Creates a MultipleChoiceTemplate instance with specific parameters.

        @param override_args: A dictionary containing override options for the template.

        @return: MultipleChoiceTemplate: The created template instance.
        """
        pass

    @abstractmethod
    def create_and_process_metadata(self, created_templates: List[Template], dataset_name: str,
                                    override_options: dict) -> None:
        """
        Creates and processes metadata for the templates.

        @param created_templates: A list of created templates.
        @param dataset_name: The name of the dataset.
        @param override_options: A dictionary containing override options for the template.
        @return: None
        """
        pass
