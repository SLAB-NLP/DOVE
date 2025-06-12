import os
import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src.plotting.utils.DataLoader import DataLoader, get_repo_files
from src.plotting.utils.config import ALL_DATASETS


class DataManager:
    """
    Centralized data management class for loading, processing, and aggregating
    model evaluation data from HuggingFace with memory-efficient handling and persistent local cache.
    """

    def __init__(self, temp_dir: Optional[str] = None, use_cache: bool = False,
                 persistent_cache_dir: Optional[str] = None):
        """
        Initialize DataManager.
        
        Args:
            temp_dir: Optional temporary directory path. If None, creates system temp dir.
            use_cache: Whether to use cached data files or always reload from HF
            persistent_cache_dir: Optional persistent cache directory for parquet files.
                                If provided, will store/load parquet files locally.
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="llm_eval_")
        self.use_cache = use_cache
        self.data_loader = DataLoader()

        # Persistent cache directory for parquet files
        self.persistent_cache_dir = persistent_cache_dir
        if self.persistent_cache_dir:
            Path(self.persistent_cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"Using persistent cache directory: {self.persistent_cache_dir}")

        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Dataset configurations - now from config file
        self.interesting_datasets = ALL_DATASETS
        self.required_columns = [
            'dataset', 'sample_index', 'model',
            'dimensions_5: shots', 'dimensions_4: instruction_phrasing_text', 'dimensions_2: separator',
            'dimensions_1: enumerator', 'dimensions_3: choices_order',
            'score'
        ]

    def load_model_data(
            self,
            model_name: str,
            shots_list: List[int] = [0, 5],
            datasets: Optional[List[str]] = None,
            num_processes: int = 4,
            force_reload: bool = False,
            repo_files: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and process data for a single model across all datasets.
        
        Args:
            model_name: Name of the model (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
            shots_list: List of shot counts to evaluate
            datasets: List of datasets to process. If None, uses all datasets.
            num_processes: Number of parallel processes for data loading
            force_reload: Force reload from HF even if cached file exists
            
        Returns:
            DataFrame with aggregated results for the model
        """
        datasets = datasets or self.interesting_datasets

        # Check for persistent cache first (if configured)
        if self.persistent_cache_dir and not force_reload:
            persistent_cache_file = self._get_persistent_cache_path(model_name, datasets, shots_list)
            if persistent_cache_file.exists():
                print(f"✅ Loading from persistent cache: {persistent_cache_file.name}")
                return pd.read_parquet(persistent_cache_file)

        # If no persistent cache available, check temporary cache
        model_filename = model_name.split('/')[-1]
        datasets_hash = hash(tuple(sorted(datasets)))  # Create hash of dataset list
        shots_hash = hash(tuple(sorted(shots_list)))  # Create hash of shots list
        cache_file = Path(self.temp_dir) / f"aggregated_{model_filename}_{datasets_hash}_{shots_hash}.parquet"

        if not force_reload and cache_file.exists() and self.use_cache:
            print(f"✅ Loading temp cached data for {model_name} (datasets: {len(datasets)}, shots: {shots_list})")
            return pd.read_parquet(cache_file)

        print(f"🔄 No cache found - downloading from HuggingFace for model: {model_name}")
        print(f"    Datasets: {datasets}")
        print(f"    Shots: {shots_list}")

        # Get repository files once for all configurations (or use provided list)
        if repo_files is None:
            repo_files = get_repo_files()

        # Get HF token for worker processes
        hf_token = os.getenv('HF_ACCESS_TOKEN')

        # Create parameter combinations with repo_files and token
        params_list = [
            (model_name, shots, dataset, repo_files, hf_token)
            for dataset in datasets
            for shots in shots_list
        ]

        # Process configurations (try parallel first, fallback to sequential)
        results = []
        try:
            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    for result in tqdm(
                            pool.imap_unordered(self._process_single_configuration, params_list),
                            total=len(params_list),
                            desc=f"Processing {model_name}"
                    ):
                        if result is not None:
                            results.append(result)
            else:
                # Sequential processing
                for params in tqdm(params_list, desc=f"Processing {model_name}"):
                    result = self._process_single_configuration(params)
                    if result is not None:
                        results.append(result)
        except Exception as e:
            print(f"⚠️  Error in parallel processing: {e}")
            print("🔄 Falling back to sequential processing...")
            results = []
            for params in tqdm(params_list, desc=f"Processing {model_name} (sequential)"):
                result = self._process_single_configuration(params)
                if result is not None:
                    results.append(result)

        if not results:
            print(f"Warning: No data loaded for model {model_name}")
            return pd.DataFrame()

        # Combine results
        combined_df = pd.concat(results, ignore_index=True)

        # Save to temporary cache
        # combined_df.to_parquet(cache_file, index=False)
        # print(f"Cached data saved for {model_name} (datasets: {len(datasets)}, shots: {shots_list}): {combined_df.shape}")

        # Save to persistent cache if configured
        if self.persistent_cache_dir:
            persistent_cache_file = self._get_persistent_cache_path(model_name, datasets, shots_list)
            combined_df.to_parquet(persistent_cache_file, index=False)
            print(f"💾 Saved to persistent cache: {persistent_cache_file.name}")

        return combined_df

    def _get_persistent_cache_path(self, model_name: str, datasets: List[str], shots_list: List[int]) -> Path:
        """
        Generate path for persistent cache file.
        
        Args:
            model_name: Name of the model
            datasets: List of datasets 
            shots_list: List of shot counts
            
        Returns:
            Path object for the cache file
        """
        model_filename = model_name.split('/')[-1]

        # If single dataset, use simple format: model_dataset_shots.parquet
        if len(datasets) == 1:
            dataset_name = datasets[0].replace('.', '_').replace('/', '_')
            shots_str = '_'.join(map(str, sorted(shots_list)))
            filename = f"{model_filename}_{dataset_name}.parquet"
        else:
            # If multiple datasets, create a combined identifier
            datasets_str = '_'.join([d.replace('.', '_').replace('/', '_') for d in sorted(datasets)])[
                           :50]  # Limit length
            shots_str = '_'.join(map(str, sorted(shots_list)))
            filename = f"{model_filename}_multi_{len(datasets)}datasets_{shots_str}shot.parquet"

        return Path(self.persistent_cache_dir) / filename

    def _process_single_configuration(self, params) -> Optional[pd.DataFrame]:
        """Process a single configuration (model, shots, dataset, repo_files, hf_token)."""
        model_name, shots, dataset, repo_files, hf_token = params

        try:
            # Silent HF authentication in worker process
            # Load raw data with pre-fetched repo files
            df = self.data_loader.load_and_process_data(
                model_name=model_name,
                shots=shots,
                datasets=[dataset],
                max_samples=None,
                repo_files=repo_files
            )

            if df.empty:
                return None

            # Filter out unwanted configurations if needed

            # Remove duplicates
            df = df.drop_duplicates(
                subset=['dimensions_5: shots', 'dimensions_4: instruction_phrasing_text', 'dimensions_2: separator',
                        'dimensions_1: enumerator',
                        'dimensions_3: choices_order', 'model', 'dataset', 'sample_index']
            )

            return df

        except Exception as e:
            error_msg = str(e)
            # Check for HF authentication errors
            if any(phrase in error_msg for phrase in ["Invalid user token", "401", "403", "unauthorized"]):
                print(f"❌ HuggingFace authentication failed: {error_msg}")
                print("💡 Please run: huggingface-cli login")
                print("   Or update your HF_ACCESS_TOKEN environment variable")
                exit(1)
            elif "Too Many Requests" in error_msg or "429" in error_msg:
                print(f"⚠️  HuggingFace rate limit exceeded: {error_msg}")
                print("💡 Please wait a few minutes and try again")
                exit(1)
            else:
                print(f"Error processing {model_name}, {shots} shots, {dataset}: {e}")
                return None

    def compute_aggregated_performance(
            self,
            data: pd.DataFrame,
            group_by_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute aggregated performance metrics from raw data.
        
        Args:
            data: Raw evaluation data
            group_by_columns: Columns to group by. If None, uses default set.
            
        Returns:
            DataFrame with aggregated performance metrics
        """
        if group_by_columns is None:
            group_by_columns = [
                'dimensions_5: shots', 'dimensions_4: instruction_phrasing_text', 'dimensions_2: separator',
                'dimensions_1: enumerator',
                'dimensions_3: choices_order', 'model', 'dataset'
            ]

        # Compute aggregated metrics
        aggregated = data.groupby(group_by_columns).agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()

        # Flatten column names
        aggregated.columns = group_by_columns + ['accuracy', 'std', 'count']

        # Filter by minimum count if needed
        # aggregated = aggregated[aggregated['count'] >= min_count]

        return aggregated

    def load_multiple_models(
            self,
            model_names: List[str],
            datasets: Optional[List[str]] = None,
            shots_list: List[int] = [0, 5],
            aggregate: bool = True,
            num_processes: int = 4
    ) -> pd.DataFrame:
        """
        Load data for multiple models.
        
        Args:
            model_names: List of model names
            datasets: List of datasets to process
            shots_list: List of shot counts
            aggregate: Whether to return aggregated performance or raw data
            num_processes: Number of processes for parallel loading
            
        Returns:
            Combined DataFrame for all models
        """
        all_model_data = []

        # Get repository files once for all models (efficiency improvement)
        repo_files = get_repo_files()

        for model_name in tqdm(model_names, desc="Loading models"):
            model_data = self.load_model_data(
                model_name=model_name,
                shots_list=shots_list,
                datasets=datasets,
                num_processes=num_processes,
                repo_files=repo_files
            )

            if not model_data.empty:
                if aggregate:
                    model_data = self.compute_aggregated_performance(model_data)
                all_model_data.append(model_data)

        if not all_model_data:
            return pd.DataFrame()

        return pd.concat(all_model_data, ignore_index=True)

    def get_dataset_specific_data(
            self,
            data: pd.DataFrame,
            dataset_name: str
    ) -> pd.DataFrame:
        """Get data for a specific dataset."""
        return data[data['dataset'] == dataset_name].copy()

    def cleanup(self):
        """Clean up temporary files and directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files."""
        self.cleanup()
