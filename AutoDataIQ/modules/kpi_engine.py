"""Automatic KPI Detection Engine for AutoDataIQ.

Phase 10: Inspects the uploaded dataset, identifies measurable indicators,
calculates their values, documents formulas, and provides metadata
(category, source columns, plain-language description, priority, caveats).

KPI Categories:
1. Overview: Dataset-wide scale, missingness, duplicates, unique entities.
2. Potential Business: Inferred revenue, profit, cost, volume, price, salary, rating.
3. Statistical Numerical: Quantitative sums, averages, medians, value ranges.
4. Statistical Categorical: Class cardinality, dominant category shares.
5. Temporal: Earliest/latest dates, time horizon, period counts.

Strictly read-only: Never modifies, imputes, or alters the input DataFrame.
"""

import re
from typing import Any
import numpy as np
import pandas as pd

from modules.analysis_engine import classify_columns_semantically
from modules.semantic_detector import detect_semantic_types

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

# Keywords to detect potential business indicators from column names
BUSINESS_PATTERNS = [
    {
        "keywords": {"revenue", "sales", "turnover", "income"},
        "concept": "Revenue",
        "sum_label": "Total Revenue (Inferred)",
        "avg_label": "Average Revenue (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"profit", "earnings", "margin", "gain", "net_profit"},
        "concept": "Profit",
        "sum_label": "Total Profit (Inferred)",
        "avg_label": "Average Profit (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"cost", "expense", "spend", "expenditure", "budget"},
        "concept": "Cost",
        "sum_label": "Total Cost / Expense (Inferred)",
        "avg_label": "Average Cost (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"quantity", "volume", "units", "qty", "items_sold", "units_sold"},
        "concept": "Volume",
        "sum_label": "Total Units / Volume (Inferred)",
        "avg_label": "Average Order Volume (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"price", "fare", "fee", "rate", "tariff", "unit_price"},
        "concept": "Price",
        "sum_label": None,  # Summing price usually does not make business sense
        "avg_label": "Average Unit Price (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"discount", "rebate"},
        "concept": "Discount",
        "sum_label": None,
        "avg_label": "Average Discount (Inferred)",
        "priority": "Medium",
    },
    {
        "keywords": {"salary", "wage", "compensation", "pay", "payroll"},
        "concept": "Compensation",
        "sum_label": "Total Payroll (Inferred)",
        "avg_label": "Average Salary (Inferred)",
        "priority": "High",
    },
    {
        "keywords": {"score", "rating", "nps", "satisfaction", "review_score"},
        "concept": "Performance Rating",
        "sum_label": None,
        "avg_label": "Average Rating / Score (Inferred)",
        "priority": "High",
    },
]

# Patterns for numerical variables where SUM() does NOT make physical sense
NON_SUMMABLE_PATTERNS = {
    "age",
    "year",
    "month",
    "day",
    "pct",
    "percent",
    "percentage",
    "ratio",
    "rate",
    "score",
    "rating",
    "gpa",
    "rank",
    "ranking",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "lng",
    "index",
    "iq",
}


def _format_kpi_value(val: Any) -> str:
    """Format KPI output values for consistent visual rendering."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        elif abs(val) >= 0.01:
            return f"{val:.2f}"
        elif val == 0:
            return "0.00"
        else:
            return f"{val:.4f}"
    return str(val)


def detect_kpis(
    dataframe: pd.DataFrame,
    semantic_types_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Inspect dataset and automatically extract prioritized, explainable KPIs.

    Parameters:
    - dataframe: The input dataset (strictly read-only).
    - semantic_types_df: Optional precomputed semantic types DataFrame.

    Returns:
    dict containing:
    - summary: counts of total, priority, and category metrics
    - overview_metrics: key dataset-level scalar metrics
    - kpis: list of individual KPI dicts
    - kpis_df: pd.DataFrame representation for table display
    """
    if dataframe is None or dataframe.empty:
        empty_df = pd.DataFrame(
            columns=[
                "KPI Name",
                "Category",
                "Value",
                "Source Column(s)",
                "Formula",
                "Description",
                "Priority",
                "Warnings",
                "Unit",
            ]
        )
        return {
            "summary": {
                "total_kpis": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0,
                "category_counts": {
                    "Overview": 0,
                    "Potential Business": 0,
                    "Statistical Numerical": 0,
                    "Statistical Categorical": 0,
                    "Temporal": 0,
                },
            },
            "overview_metrics": {
                "total_records": 0,
                "total_columns": 0,
                "missing_cells": 0,
                "missing_percentage": 0.0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "unique_entities": {},
            },
            "kpis": [],
            "kpis_df": empty_df,
        }

    # 1. Resolve semantic column groupings
    if semantic_types_df is None or semantic_types_df.empty:
        semantic_types_df = detect_semantic_types(dataframe)

    classified_cols = classify_columns_semantically(dataframe, semantic_types_df)

    kpi_list: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def add_kpi(
        kpi_name: str,
        value: str,
        raw_value: Any,
        category: str,
        source_columns: list[str],
        formula: str,
        description: str,
        priority: str,
        warnings: str = "",
        unit: str = "",
    ) -> None:
        key = f"{category}::{kpi_name}::{'|'.join(sorted(source_columns))}"
        if key in seen_keys:
            return
        seen_keys.add(key)
        kpi_list.append(
            {
                "KPI Name": kpi_name,
                "Value": value,
                "Raw Value": raw_value,
                "Category": category,
                "Source Column(s)": source_columns,
                "Formula": formula,
                "Description": description,
                "Priority": priority,
                "Warnings": warnings,
                "Unit": unit,
            }
        )

    # -------------------------------------------------------------
    # 2. Dataset Overview KPIs
    # -------------------------------------------------------------
    overview_metrics = _extract_overview_kpis(
        dataframe, classified_cols, add_kpi
    )

    # -------------------------------------------------------------
    # 3. Potential Business KPIs (Inferred from Semantics & Headers)
    # -------------------------------------------------------------
    business_cols = _extract_potential_business_kpis(
        dataframe, classified_cols, add_kpi
    )

    # -------------------------------------------------------------
    # 4. Statistical Numerical KPIs (Standard Quantitative Metrics)
    # -------------------------------------------------------------
    _extract_numerical_kpis(
        dataframe, classified_cols, business_cols, add_kpi
    )

    # -------------------------------------------------------------
    # 5. Statistical Categorical KPIs
    # -------------------------------------------------------------
    _extract_categorical_kpis(dataframe, classified_cols, add_kpi)

    # -------------------------------------------------------------
    # 6. Temporal Date-Based KPIs
    # -------------------------------------------------------------
    _extract_temporal_kpis(dataframe, classified_cols, add_kpi)

    # -------------------------------------------------------------
    # 7. Priority Sorting & Summary Compilation
    # -------------------------------------------------------------
    kpi_list.sort(key=lambda x: PRIORITY_ORDER.get(x["Priority"], 99))

    high_count = sum(1 for k in kpi_list if k["Priority"] == "High")
    med_count = sum(1 for k in kpi_list if k["Priority"] == "Medium")
    low_count = sum(1 for k in kpi_list if k["Priority"] == "Low")

    cat_counts: dict[str, int] = {
        "Overview": 0,
        "Potential Business": 0,
        "Statistical Numerical": 0,
        "Statistical Categorical": 0,
        "Temporal": 0,
    }
    for k in kpi_list:
        cat = k["Category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    summary = {
        "total_kpis": len(kpi_list),
        "high_priority": high_count,
        "medium_priority": med_count,
        "low_priority": low_count,
        "category_counts": cat_counts,
    }

    # Format table for UI rendering
    table_rows = []
    for item in kpi_list:
        table_rows.append(
            {
                "KPI Name": item["KPI Name"],
                "Category": item["Category"],
                "Value": str(item["Value"]),
                "Source Column(s)": ", ".join(item["Source Column(s)"])
                if item["Source Column(s)"]
                else "Dataset-wide",
                "Formula": item["Formula"],
                "Description": item["Description"],
                "Priority": item["Priority"],
                "Warnings": item["Warnings"] if item["Warnings"] else "None",
                "Unit": item["Unit"] if item["Unit"] else "-",
            }
        )

    kpis_df = pd.DataFrame(
        table_rows,
        columns=[
            "KPI Name",
            "Category",
            "Value",
            "Source Column(s)",
            "Formula",
            "Description",
            "Priority",
            "Warnings",
            "Unit",
        ],
    )

    return {
        "summary": summary,
        "overview_metrics": overview_metrics,
        "kpis": kpi_list,
        "kpis_df": kpis_df,
    }


def _extract_overview_kpis(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_kpi: Any,
) -> dict[str, Any]:
    """Compute dataset-level overview indicators and return scalar summary."""
    row_count, col_count = dataframe.shape
    total_cells = row_count * col_count
    missing_cells = int(dataframe.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0

    dup_count = int(dataframe.duplicated().sum())
    dup_pct = (dup_count / row_count * 100.0) if row_count > 0 else 0.0

    # Total Records
    add_kpi(
        kpi_name="Total Records",
        value=f"{row_count:,}",
        raw_value=row_count,
        category="Overview",
        source_columns=[],
        formula="COUNT(rows)",
        description="Total number of observations / rows in the uploaded dataset.",
        priority="High",
        warnings="",
        unit="records",
    )

    # Total Columns / Features
    add_kpi(
        kpi_name="Total Attributes",
        value=f"{col_count:,}",
        raw_value=col_count,
        category="Overview",
        source_columns=[],
        formula="COUNT(columns)",
        description="Total number of fields / attributes defined in the schema.",
        priority="Medium",
        warnings="",
        unit="attributes",
    )

    # Missing Value Rate
    miss_prio = "High" if missing_pct >= 10.0 else "Low"
    miss_warn = (
        "Dataset contains substantial missing values that may skew aggregations."
        if missing_pct >= 10.0
        else ""
    )
    add_kpi(
        kpi_name="Overall Missing Value Rate",
        value=f"{missing_pct:.1f}%",
        raw_value=missing_pct,
        category="Overview",
        source_columns=[],
        formula="SUM(ISNULL(cells)) / (ROWS * COLS) * 100",
        description=f"Percentage of unpopulated / null cells across all features ({missing_cells:,} missing cells).",
        priority=miss_prio,
        warnings=miss_warn,
        unit="%",
    )

    # Duplicate Records
    dup_prio = "High" if dup_pct >= 5.0 else ("Medium" if dup_count > 0 else "Low")
    dup_warn = (
        "Duplicate rows detected; consider deduplication to avoid double counting."
        if dup_count > 0
        else ""
    )
    add_kpi(
        kpi_name="Duplicate Row Count",
        value=f"{dup_count:,} ({dup_pct:.1f}%)",
        raw_value=dup_count,
        category="Overview",
        source_columns=[],
        formula="COUNT(duplicate_rows)",
        description=f"Exact duplicate records identified across the dataset ({dup_pct:.1f}% duplicate rate).",
        priority=dup_prio,
        warnings=dup_warn,
        unit="rows",
    )

    # Unique Entities from Identifier Columns
    unique_entities: dict[str, int] = {}
    id_cols = classified_cols.get("identifier", [])
    for id_col in id_cols:
        series = dataframe[id_col].dropna()
        n_unique = int(series.nunique())
        unique_entities[id_col] = n_unique
        n_valid = len(series)
        uniq_pct = (n_unique / n_valid * 100.0) if n_valid > 0 else 0.0

        add_kpi(
            kpi_name=f"Distinct Entities ({id_col})",
            value=f"{n_unique:,}",
            raw_value=n_unique,
            category="Overview",
            source_columns=[id_col],
            formula=f"COUNT(DISTINCT {id_col})",
            description=f"Number of distinct entities identified by primary key column '{id_col}' ({uniq_pct:.1f}% uniqueness).",
            priority="High" if uniq_pct >= 95.0 else "Medium",
            warnings="Exempted from arithmetic aggregations to preserve semantic integrity.",
            unit="entities",
        )

    return {
        "total_records": row_count,
        "total_columns": col_count,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_pct, 2),
        "duplicate_rows": dup_count,
        "duplicate_percentage": round(dup_pct, 2),
        "unique_entities": unique_entities,
    }


def _extract_potential_business_kpis(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_kpi: Any,
) -> set[str]:
    """Detect potential business KPIs inferred from column names and numerical semantics."""
    num_cols = classified_cols.get("numerical", [])
    identified_business_cols: set[str] = set()

    for col in num_cols:
        col_lower = str(col).lower()
        series = pd.to_numeric(dataframe[col], errors="coerce").dropna()
        if series.empty:
            continue

        n_valid = len(series)
        total_rows = len(dataframe)
        miss_count = total_rows - n_valid
        miss_pct = (miss_count / total_rows * 100.0) if total_rows > 0 else 0.0

        # Match against business patterns
        for pattern in BUSINESS_PATTERNS:
            keywords = pattern["keywords"]
            # Check if any keyword matches as a token or substring
            matches = any(kw in col_lower for kw in keywords)
            if matches:
                identified_business_cols.add(col)
                concept = pattern["concept"]
                sum_label = pattern["sum_label"]
                avg_label = pattern["avg_label"]
                base_priority = pattern["priority"]

                # Caveat string
                caveat = (
                    f"Inferred {concept} indicator based on column name '{col}'. "
                    "Commercial currency or exact business unit is unconfirmed; verify with domain schema."
                )
                if miss_pct >= 10.0:
                    caveat += f" Note: {miss_pct:.1f}% missing values ({miss_count:,} records)."

                # 1. Potential Total Sum KPI
                if sum_label is not None:
                    sum_val = float(series.sum())
                    add_kpi(
                        kpi_name=f"{sum_label} [{col}]",
                        value=_format_kpi_value(sum_val),
                        raw_value=sum_val,
                        category="Potential Business",
                        source_columns=[col],
                        formula=f"SUM({col})",
                        description=f"Inferred aggregated {concept.lower()} total calculated across {n_valid:,} valid entries in '{col}'.",
                        priority=base_priority,
                        warnings=caveat,
                        unit="",
                    )

                # 2. Potential Average / Mean KPI
                if avg_label is not None:
                    mean_val = float(series.mean())
                    add_kpi(
                        kpi_name=f"{avg_label} [{col}]",
                        value=_format_kpi_value(mean_val),
                        raw_value=mean_val,
                        category="Potential Business",
                        source_columns=[col],
                        formula=f"AVG({col})",
                        description=f"Inferred average {concept.lower()} measure per record across {n_valid:,} valid entries in '{col}'.",
                        priority=base_priority,
                        warnings=caveat,
                        unit="",
                    )

                # Stop after matching the best pattern for this column
                break

    return identified_business_cols


def _extract_numerical_kpis(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    business_cols: set[str],
    add_kpi: Any,
) -> None:
    """Compute standard statistical KPIs for numerical features."""
    num_cols = classified_cols.get("numerical", [])

    for col in num_cols:
        series = pd.to_numeric(dataframe[col], errors="coerce").dropna()
        if series.empty:
            continue

        n_valid = len(series)
        total_rows = len(dataframe)
        miss_count = total_rows - n_valid
        miss_pct = (miss_count / total_rows * 100.0) if total_rows > 0 else 0.0

        col_lower = str(col).lower()
        is_summable = not any(pat in col_lower for pat in NON_SUMMABLE_PATTERNS)

        mean_val = float(series.mean())
        median_val = float(series.median())
        min_val = float(series.min())
        max_val = float(series.max())

        warn = f"Calculated from {n_valid:,} valid records."
        if miss_pct > 0:
            warn += f" ({miss_pct:.1f}% missing values)."

        # Priority: If column was not already highlighted as a business KPI, give it Medium/Low
        prio = "Medium" if col not in business_cols else "Low"

        # 1. Mean KPI
        add_kpi(
            kpi_name=f"Mean ({col})",
            value=_format_kpi_value(mean_val),
            raw_value=mean_val,
            category="Statistical Numerical",
            source_columns=[col],
            formula=f"AVG({col})",
            description=f"Arithmetic average value for numerical feature '{col}'.",
            priority=prio,
            warnings=warn,
            unit="",
        )

        # 2. Median KPI
        add_kpi(
            kpi_name=f"Median ({col})",
            value=_format_kpi_value(median_val),
            raw_value=median_val,
            category="Statistical Numerical",
            source_columns=[col],
            formula=f"MEDIAN({col})",
            description=f"50th percentile (median) value for '{col}', robust against extreme tail skew.",
            priority="Low",
            warnings=warn,
            unit="",
        )

        # 3. Sum KPI (only when summable and not redundant with business KPI)
        if is_summable and col not in business_cols:
            sum_val = float(series.sum())
            add_kpi(
                kpi_name=f"Sum ({col})",
                value=_format_kpi_value(sum_val),
                raw_value=sum_val,
                category="Statistical Numerical",
                source_columns=[col],
                formula=f"SUM({col})",
                description=f"Total aggregate sum of all valid observations in '{col}'.",
                priority=prio,
                warnings=warn,
                unit="",
            )

        # 4. Range KPI (Min - Max)
        add_kpi(
            kpi_name=f"Range ({col})",
            value=f"[{_format_kpi_value(min_val)} – {_format_kpi_value(max_val)}]",
            raw_value=max_val - min_val,
            category="Statistical Numerical",
            source_columns=[col],
            formula=f"[MIN({col}), MAX({col})]",
            description=f"Observed value boundaries from minimum ({_format_kpi_value(min_val)}) to maximum ({_format_kpi_value(max_val)}).",
            priority="Low",
            warnings=warn,
            unit="",
        )


def _extract_categorical_kpis(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_kpi: Any,
) -> None:
    """Compute cardinality and dominance indicators for categorical features."""
    cat_cols = classified_cols.get("categorical", [])
    bool_cols = classified_cols.get("boolean", [])

    # Categorical Columns
    for col in cat_cols:
        series = dataframe[col].dropna().astype(str)
        if series.empty:
            continue

        n_valid = len(series)
        counts = series.value_counts()
        n_unique = len(counts)

        # 1. Distinct Classes Count
        add_kpi(
            kpi_name=f"Distinct Classes ({col})",
            value=f"{n_unique:,}",
            raw_value=n_unique,
            category="Statistical Categorical",
            source_columns=[col],
            formula=f"COUNT(DISTINCT {col})",
            description=f"Count of distinct unique categorical groupings in '{col}'.",
            priority="Medium",
            warnings="",
            unit="classes",
        )

        # 2. Dominant Category Proportion
        top_cat = str(counts.index[0])
        top_cnt = int(counts.iloc[0])
        top_pct = (top_cnt / n_valid * 100.0) if n_valid > 0 else 0.0

        prio = "High" if top_pct >= 70.0 else "Medium"
        warn = (
            f"Class '{top_cat}' constitutes majority share ({top_pct:.1f}%)."
            if top_pct >= 70.0
            else ""
        )
        add_kpi(
            kpi_name=f"Dominant Category ({col})",
            value=f"'{top_cat}' ({top_pct:.1f}%)",
            raw_value=top_pct,
            category="Statistical Categorical",
            source_columns=[col],
            formula=f"MODE({col}) with COUNT / TOTAL * 100",
            description=f"Most frequent category '{top_cat}' accounts for {top_cnt:,} of {n_valid:,} observations.",
            priority=prio,
            warnings=warn,
            unit="%",
        )

    # Boolean Columns
    for col in bool_cols:
        series = dataframe[col].dropna()
        if series.empty:
            continue

        n_valid = len(series)
        truthy_vals = {"true", "1", "t", "yes", "y"}
        true_cnt = sum(1 for v in series if str(v).strip().lower() in truthy_vals)
        true_pct = (true_cnt / n_valid * 100.0) if n_valid > 0 else 0.0

        add_kpi(
            kpi_name=f"Positive Rate ({col})",
            value=f"{true_pct:.1f}% ({true_cnt:,}/{n_valid:,})",
            raw_value=true_pct,
            category="Statistical Categorical",
            source_columns=[col],
            formula=f"SUM(CASE WHEN {col}=True THEN 1 ELSE 0 END) / COUNT({col}) * 100",
            description=f"Proportion of records with positive (True) status in boolean indicator '{col}'.",
            priority="Medium",
            warnings="",
            unit="%",
        )


def _extract_temporal_kpis(
    dataframe: pd.DataFrame,
    classified_cols: dict[str, list[str]],
    add_kpi: Any,
) -> None:
    """Compute temporal range and period density indicators for date features."""
    date_cols = classified_cols.get("date", [])
    if not date_cols:
        return

    import warnings

    for col in date_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(dataframe[col], errors="coerce", format="mixed")

        valid_dates = parsed.dropna()
        if valid_dates.empty:
            continue

        min_dt = valid_dates.min()
        max_dt = valid_dates.max()
        days_span = (max_dt - min_dt).days

        # 1. Date Span KPI
        add_kpi(
            kpi_name=f"Temporal Range ({col})",
            value=f"{min_dt.date()} to {max_dt.date()} ({days_span:,} days)",
            raw_value=days_span,
            category="Temporal",
            source_columns=[col],
            formula=f"MIN({col}) to MAX({col})",
            description=f"Earliest and latest recorded dates in '{col}' spanning {days_span:,} days.",
            priority="High",
            warnings=f"Computed from {len(valid_dates):,} valid date records.",
            unit="days",
        )

        # 2. Distinct Monthly Periods
        months = valid_dates.dt.to_period("M").nunique()
        add_kpi(
            kpi_name=f"Distinct Months ({col})",
            value=f"{months:,} months",
            raw_value=months,
            category="Temporal",
            source_columns=[col],
            formula=f"COUNT(DISTINCT TO_PERIOD({col}, 'M'))",
            description=f"Number of distinct calendar month intervals represented across '{col}'.",
            priority="Medium",
            warnings="",
            unit="months",
        )
