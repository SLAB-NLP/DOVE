"""
DataLoader module for loading and processing model evaluation data from HuggingFace DOVE dataset.

This module provides functionality to load, filter, and process model evaluation data
with efficient memory management and caching capabilities.
"""

import os
import time
from typing import List, Optional, Union

import pandas as pd
import pyarrow.compute as pc
from datasets import load_dataset
from huggingface_hub import HfApi

# Repository configuration
REPO_NAME = "nlphuji/DOVE_Lite"

# Global repository files cache to avoid multiple API calls
_repo_files_cache = None

def get_repo_files(force_refresh: bool = False) -> List[str]:
    """
    Get list of parquet files in the HuggingFace repository.
    
    Uses caching to avoid multiple API calls per session.
    
    Args:
        force_refresh: Force refresh the cache even if it exists
        
    Returns:
        List of parquet files in the repository
        
    Raises:
        SystemExit: If rate limit exceeded or authentication fails
    """
    global _repo_files_cache
    
    if _repo_files_cache is None or force_refresh:
        print("📂 Fetching repository files from HuggingFace (one-time operation)...")
        try:
            api = HfApi()
            all_files = api.list_repo_files(
                repo_id=REPO_NAME,
                repo_type="dataset",
            )
            _repo_files_cache = [file for file in all_files if file.endswith('.parquet')]
            print(f"✅ Found {len(_repo_files_cache)} parquet files in repository")
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "429" in error_msg:
                print(f"⚠️  HuggingFace rate limit exceeded while fetching files: {error_msg}")
                print("💡 Please wait a few minutes and try again")
                exit(1)
            else:
                print(f"❌ Error fetching repository files: {error_msg}")
                exit(1)
    
    return _repo_files_cache


class DataLoader:
    """
    DataLoader for HuggingFace DOVE dataset with efficient filtering and processing.
    
    This class provides methods to load and process model evaluation data with
    memory-efficient filtering and batch processing capabilities.
    """

    def __init__(self, dataset_name: str = REPO_NAME, split: str = "train", batch_size: int = 10000):
        """
        Initialize the DataLoader with dataset configuration.

        Args:
            dataset_name: Name of the dataset to load from HuggingFace
            split: Dataset split to use (default: "train")
            batch_size: Number of samples per batch for processing
        """
        self.dataset = None
        self.repo_name = dataset_name
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(
        self, 
        model_name: str, 
        shots: Optional[int] = None,
        datasets: Optional[Union[str, List[str]]] = None, 
        template: Optional[str] = None, 
        separator: Optional[str] = None, 
        enumerator: Optional[str] = None, 
        choices_order: Optional[str] = None,
        max_samples: Optional[int] = None, 
        drop_columns: bool = True, 
        repo_files: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and process data for specified model and configuration.
        
        Args:
            model_name: Name of the model to load data for
            shots: Number of shots (few-shot learning examples)
            datasets: Dataset name(s) to filter by
            template: Template configuration to filter by
            separator: Separator configuration to filter by
            enumerator: Enumerator configuration to filter by
            choices_order: Choices order configuration to filter by
            max_samples: Maximum number of samples to load
            drop_columns: Whether to drop unnecessary columns for memory efficiency
            repo_files: Pre-fetched list of repository files for efficiency
            
        Returns:
            Processed DataFrame with the requested data
        """
        start_time = time.time()
        
        self.load_data_with_filter(
            max_samples=max_samples, 
            drop_columns=drop_columns, 
            model_name=model_name, 
            shots=shots, 
            datasets=datasets, 
            repo_files=repo_files
        )
        
        if self.dataset is None or len(self.dataset) == 0:
            print(f"No data found for model {model_name} and shots {shots}")
            return pd.DataFrame()
        
        load_time = time.time()
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return self.dataset.to_pandas()

    def load_data_with_filter(
        self, 
        max_samples: Optional[int] = None, 
        drop_columns: bool = True, 
        model_name: Optional[str] = None, 
        shots: Optional[int] = None, 
        datasets: Optional[Union[str, List[str]]] = None, 
        repo_files: Optional[List[str]] = None
    ) -> None:
        """
        Efficiently load data with filtering during the loading process.
        
        Uses pyarrow for memory-efficient filtering and processing.
        
        Args:
            max_samples: Maximum number of samples to load
            drop_columns: Whether to drop unnecessary columns to save memory
            model_name: Model name to filter by
            shots: Number of shots to filter by
            datasets: Dataset name(s) to filter by
            repo_files: Pre-fetched list of repository files for efficiency
        """
        # Get repository files (use provided list or fetch from cache/API)
        if repo_files is not None:
            existing_files = repo_files
        else:
            existing_files = get_repo_files()
        
        # Apply filters to file list
        existing_files = self._filter_files_by_criteria(
            existing_files, model_name, shots, datasets
        )

        if len(existing_files) > 0:
            try:
                self.dataset = load_dataset(
                    self.repo_name, 
                    data_files=existing_files, 
                    split=self.split,
                    keep_in_memory=True,
                    cache_dir=None
                )
                print(f"Loaded dataset with {len(self.dataset)} samples")
                
                if drop_columns and self.dataset is not None:
                    self._drop_unnecessary_columns()
                    
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.dataset = None
        else:
            print("No data files found matching the specified criteria.")
            self.dataset = None

    def _filter_files_by_criteria(
        self, 
        files: List[str], 
        model_name: Optional[str], 
        shots: Optional[int], 
        datasets: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """
        Filter repository files based on specified criteria.
        
        Args:
            files: List of all available files
            model_name: Model name to filter by
            shots: Number of shots to filter by
            datasets: Dataset name(s) to filter by
            
        Returns:
            Filtered list of files matching the criteria
        """
        filtered_files = files.copy()
        
        # Filter by model name
        if model_name is not None:
            model_short_name = model_name.split("/")[-1]
            filtered_files = [f for f in filtered_files if model_short_name in f]
        
        # Filter by shots
        if shots is not None:
            shots_str = f"{shots}_shot"
            filtered_files = [f for f in filtered_files if shots_str in f]
        
        # Filter by datasets
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]
            dataset_names = [dataset.split("/")[-1] for dataset in datasets]
            filtered_files = [
                f for f in filtered_files 
                if any(dataset_name in f for dataset_name in dataset_names)
            ]
        
        return filtered_files

    def _drop_unnecessary_columns(self) -> None:
        """
        Drop columns that are not needed for analysis to save memory.
        """
        columns_to_drop = [
            'raw_input', 
            'generated_text', 
            'cumulative_logprob', 
            'closest_answer', 
            'ground_truth'
        ]
        
        # Only drop columns that actually exist in the dataset
        existing_columns = set(self.dataset.column_names)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]
        
        if columns_to_drop:
            self.dataset = self.dataset.remove_columns(columns_to_drop)

    def extract_data(
        self, 
        model_name: str, 
        shots: int, 
        dataset: Optional[str] = None, 
        template: Optional[str] = None, 
        separator: Optional[str] = None, 
        enumerator: Optional[str] = None,
        choices_order: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract filtered data based on specified criteria using PyArrow.
        
        Args:
            model_name: Model name to filter by
            shots: Number of shots to filter by
            dataset: Dataset name to filter by
            template: Template configuration to filter by
            separator: Separator configuration to filter by
            enumerator: Enumerator configuration to filter by
            choices_order: Choices order configuration to filter by
            
        Returns:
            Filtered DataFrame
            
        Raises:
            Exception: If filtering operation fails
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data_with_filter first.")
            
        arrow_table = self.dataset.data.table

        # Build filter conditions
        conditions = [
            pc.equal(arrow_table['model'], model_name),
            pc.equal(arrow_table['shots'], shots)
        ]

        # Add optional filters
        optional_filters = {
            'template': template,
            'separator': separator,
            'enumerator': enumerator,
            'choices_order': choices_order
        }

        for column, value in optional_filters.items():
            if value is not None:
                conditions.append(pc.equal(arrow_table[column], value))

        # Combine all conditions
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = pc.and_(combined_condition, condition)

        try:
            print("Filtering data with PyArrow...")
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()
            print(f"Filtered data shape: {df_filtered.shape}")
            return df_filtered
        except Exception as e:
            raise Exception(f"Error filtering data: {str(e)}")

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame based on key columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        key_columns = [
            'sample_index', 'model', 'dataset', 'template', 
            'separator', 'enumerator', 'choices_order', 'shots'
        ]
        
        # Only use columns that exist in the DataFrame
        existing_key_columns = [col for col in key_columns if col in df.columns]
        
        if not existing_key_columns:
            print("Warning: No key columns found for duplicate removal")
            return df
            
        df_unique = df.drop_duplicates(subset=existing_key_columns)
        
        removed_count = len(df) - len(df_unique)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows")
            
        return df_unique
