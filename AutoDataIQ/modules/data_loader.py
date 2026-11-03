"""Utilities for loading uploaded datasets without changing their contents."""

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when an uploaded dataset cannot be loaded safely."""


def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Load a Streamlit uploaded CSV or Excel file into a Pandas DataFrame.

    The file is read as supplied. No cleaning, type conversion, or other data
    modification is performed by this function.
    """
    if uploaded_file is None:
        raise DatasetLoadError("Please choose a dataset file before loading.")

    file_name = getattr(uploaded_file, "name", "")
    extension = Path(file_name).suffix.lower()

    if extension not in {".csv", ".xlsx", ".xls"}:
        raise DatasetLoadError(
            "Unsupported file type. Please upload a CSV, XLSX, or XLS file."
        )

    try:
        if uploaded_file.size == 0:
            raise DatasetLoadError("The uploaded file is empty.")
    except AttributeError:
        # Streamlit UploadedFile objects provide ``size``. This keeps the
        # loader usable with compatible file-like objects as well.
        pass

    try:
        uploaded_file.seek(0)
        if extension == ".csv":
            dataframe = pd.read_csv(uploaded_file)
        else:
            dataframe = pd.read_excel(uploaded_file)
    except pd.errors.EmptyDataError as error:
        raise DatasetLoadError("The uploaded file is empty or contains no readable data.") from error
    except (ValueError, OSError, UnicodeDecodeError, ImportError) as error:
        raise DatasetLoadError(
            "The file could not be read. It may be invalid, corrupted, or not match its extension."
        ) from error
    except Exception as error:
        raise DatasetLoadError("An unexpected error occurred while reading the file.") from error

    if dataframe.empty:
        raise DatasetLoadError("The dataset is empty. Please upload a file with at least one row of data.")

    return dataframe
