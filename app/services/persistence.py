"""
Dataset persistence service.
Saves uploaded dataset DataFrames into dedicated SQLite databases and supports appending.
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path

# Directory where individual dataset SQLite databases are stored
DATABASES_DIR = Path("databases")


def get_dataset_db_path(filename: str) -> Path:
    """
    Generate the database path for a given dataset filename.
    """
    # Create the databases folder if it doesn't exist
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)

    # Use the filename prefix and add .db extension
    base_name = Path(filename).stem
    # Replace spaces and special characters with underscores to keep filename safe
    safe_base_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)
    
    return DATABASES_DIR / f"{safe_base_name}.db"


def save_to_dataset_db(df: pd.DataFrame, filename: str, mode: str = "append") -> str:
    """
    Write or append the DataFrame to a table named 'data' inside a dedicated SQLite database.

    Args:
        df: The Pandas DataFrame.
        filename: Name of the uploaded dataset file.
        mode: How to write the data if table exists ('fail', 'replace', 'append').

    Returns:
        The string path of the database.
    """
    db_path = get_dataset_db_path(filename)
    
    # Establish connection and write the data
    conn = sqlite3.connect(db_path)
    try:
        # Write/append the dataframe to the table named 'data'
        df.to_sql(name="data", con=conn, if_exists=mode, index=False)
    finally:
        conn.close()

    # Return standard forward-slash path string relative to project workspace
    return str(db_path.as_posix())


def read_from_dataset_db(db_path_str: str) -> pd.DataFrame:
    """
    Read the entire dataset back from its dedicated database.

    Args:
        db_path_str: Path to the SQLite database.

    Returns:
        Pandas DataFrame containing all rows.
    """
    conn = sqlite3.connect(db_path_str)
    try:
        df = pd.read_sql_query("SELECT * FROM data", conn)
        return df
    finally:
        conn.close()
