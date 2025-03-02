import json
from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Any

import pandas as pd
import pyarrow.parquet as pq
from jsonschema import validate, ValidationError
from tqdm import tqdm


class SchemaValidator:
    """Schema validator for JSON, CSV and Parquet files
    
    This validator checks if data files conform to a given JSON schema. It supports:
    - JSON files (single object or array of objects)
    - CSV files (converted to JSON-like objects)
    - Parquet files (converted to JSON-like objects)
    
    The validator can handle both file paths and direct data input, and provides
    detailed validation results including error messages and statistics.
    """

    def __init__(self, schema_path: Optional[Union[str, Path]] = None, schema_data: Optional[Dict] = None) -> None:
        """Initialize with schema file path or schema data object

        Args:
            schema_path: Path to a JSON schema file
            schema_data: Dictionary containing a JSON schema

        Raises:
            ValueError: If neither schema_path nor schema_data is provided
        """
        if schema_path:
            schema_path = Path(schema_path)
            with schema_path.open('r') as file:
                self.schema = json.load(file)
        elif schema_data:
            self.schema = schema_data
        else:
            raise ValueError("Either schema_path or schema_data must be provided")

        # Basic schema validation
        if not isinstance(self.schema, dict):
            raise ValueError("Schema must be a dictionary")
        if "properties" not in self.schema:
            raise ValueError("Schema must contain 'properties' field")

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Union[str, Dict]]:
        """Validate a file based on its extension (.json, .csv, .parquet)

        Args:
            file_path: Path to the file to validate

        Returns:
            A tuple containing:
            - bool: Whether the file is valid according to the schema
            - Union[str, Dict]: Error message (str) or validation results (Dict)
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"

        ext = path.suffix.lower()

        if ext == '.json':
            return self._validate_json(path)
        elif ext == '.csv':
            return self._validate_tabular(pd.read_csv(path))
        elif ext == '.parquet':
            return self._validate_tabular(pq.read_table(path).to_pandas())
        else:
            return False, f"Unsupported format: {ext}. Use .json, .csv, or .parquet"

    def _validate_json(self, json_path: Path) -> Tuple[bool, str]:
        """Validate JSON file against schema

        Args:
            json_path: Path to the JSON file

        Returns:
            Tuple containing validation result (bool) and message (str)
        """
        try:
            with json_path.open('r') as file:
                data = json.load(file)

            if isinstance(data, list):
                for item in tqdm(data, desc="Validating records"):
                    validate(instance=item, schema=self.schema)
            else:
                validate(instance=data, schema=self.schema)

            return True, "JSON is valid according to the schema."

        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            return False, f"Error: {str(e)}"

    def _validate_tabular(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate tabular data (DataFrame) against schema

        Args:
            df: Pandas DataFrame containing the data to validate

        Returns:
            Tuple containing validation result (bool) and detailed results (Dict)
        """
        # Check required columns
        required_fields = self.schema.get('required', [])
        missing_columns = [field for field in required_fields if field not in df.columns]

        if missing_columns:
            return False, {
                "message": f"Missing required columns: {', '.join(missing_columns)}",
                "valid_count": 0,
                "invalid_count": df.shape[0],
                "errors": [f"Missing columns: {', '.join(missing_columns)}"]
            }

        # Convert JSON strings to objects
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith('{') else x
                    )
                except json.JSONDecodeError:
                    pass

        # Validate rows
        valid_count = 0
        invalid_count = 0
        errors = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating records"):
            try:
                # Create JSON record from row
                record = {}
                for key in self.schema.get('properties', {}):
                    if key in row:
                        record[key] = row[key]

                validate(instance=record, schema=self.schema)
                valid_count += 1
            except Exception as e:
                invalid_count += 1
                if len(errors) < 10:  # Limit stored errors
                    errors.append(f"Row {index}: {str(e)}")

        is_valid = invalid_count == 0
        results = {
            "message": "All records valid." if is_valid else
            f"{invalid_count} invalid records out of {df.shape[0]}." +
            (" Showing first 10 errors." if len(errors) > 10 else ""),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "errors": errors[:10]
        }

        return is_valid, results

    @staticmethod
    def get_summary(result: Tuple[bool, Union[str, Dict]]) -> Dict[str, Any]:
        """Get a simplified summary of validation results

        Args:
            result: The tuple returned by validation methods

        Returns:
            Dictionary with standardized summary information
        """
        is_valid, details = result

        if isinstance(details, str):
            return {
                "valid": is_valid,
                "message": details,
                "first_error": None if is_valid else details
            }
        else:
            return {
                "valid": is_valid,
                "message": details.get("message", ""),
                "valid_count": details.get("valid_count", 0),
                "invalid_count": details.get("invalid_count", 0),
                "first_error": None if is_valid or not details.get("errors") else details["errors"][0]
            }


# Example usage
if __name__ == "__main__":
    # Example paths (convert to Path objects)
    schema_file = Path("schema.json")
    test_files = {
        "json": Path("data_sample.json"),
        "csv": Path("data_sample.csv"),
        "parquet": Path("data_sample.parquet")
    }

    # Create validator
    validator = SchemaValidator(schema_path=schema_file)

    # Validate each file
    for file_type, file_path in test_files.items():
        print(f"\nValidating {file_type.upper()} file:")
        result = validator.validate_file(file_path)
        summary = SchemaValidator.get_summary(result)

        print(f"Valid: {summary['valid']}")
        print(summary['message'])

        if summary['first_error']:
            print(f"First error: {summary['first_error']}")
