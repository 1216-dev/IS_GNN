"""Data loading utilities for NYC open datasets."""

from __future__ import annotations

import pandas as pd


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_demographics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_cols(df)


def load_acs(path: str) -> pd.DataFrame:
    # Requires openpyxl; installed via requirements.
    df = pd.read_excel(path, engine="openpyxl")
    return _clean_cols(df)


def load_incidents(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_cols(df)


def load_facilities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_cols(df)


def load_income(path: str) -> pd.DataFrame:
    """Income distribution by decile/AGI range."""
    df = pd.read_csv(path)
    return _clean_cols(df)


def load_race_state(path: str) -> pd.DataFrame:
    """CVAP-style race demographics by state/region."""
    df = pd.read_csv(path)
    return _clean_cols(df)


def load_zip_csv(path: str) -> pd.DataFrame:
    """Optional helper to load a CSV inside a zip without extraction."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        first_csv = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not first_csv:
            raise ValueError(f"No CSV found in {path}")
        with zf.open(first_csv[0]) as fh:
            df = pd.read_csv(fh)
    return _clean_cols(df)
