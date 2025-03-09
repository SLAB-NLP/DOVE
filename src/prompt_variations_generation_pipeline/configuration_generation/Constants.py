from pathlib import Path


class Constants:
    """Constants used in template generation."""

    # Paths
    DATA_PATH = Path(__file__).parents[3] / "Data"
    CATALOG_PATH = DATA_PATH / "Catalog"

