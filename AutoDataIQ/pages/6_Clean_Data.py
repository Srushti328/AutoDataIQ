"""Streamlit page for Phase 11: Automated Data Cleaning and Transformation."""

import io
import pandas as pd
import streamlit as st
from typing import Any

from modules.cleaner import (
    apply_cleaning_pipeline,
    detect_cleaning_opportunities,
)
from modules.cleaning_engine import generate_cleaning_recommendations
from modules.quality_engine import analyze_data_quality
from modules.semantic_detector import detect_semantic_types

st.set_page_config(
    page_title="Automated Data Cleaning - AutoDataIQ",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Automated Data Cleaning & Transformation")
st.write(
    "Review, configure, preview, and apply robust cleaning operations to a dedicated copy of "
    "your uploaded dataset. Your original raw dataset is **never mutated**."
)

st.info(
    "🔒 **Data Protection Guarantee:** All transformations execute strictly on an isolated DataFrame copy. "
    "The original dataset remains immutable and available throughout your session."
)

# -------------------------------------------------------------
# Verify Session State Dataset
# -------------------------------------------------------------
if "data" not in st.session_state or st.session_state["data"] is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Upload page first.")
    st.info("💡 Navigate to **Upload** using the sidebar on the left.")
else:
    # Ensure raw_data is safely preserved
    if "raw_data" not in st.session_state or st.session_state["raw_data"] is None:
        st.session_state["raw_data"] = st.session_state["data"].copy(deep=True)
        st.session_state["raw_filename"] = st.session_state.get("filename", "Uploaded Dataset")

    raw_df: pd.DataFrame = st.session_state["raw_data"]
    raw_filename = st.session_state.get("raw_filename", "Uploaded Dataset")

    # -------------------------------------------------------------
    # 1. Dataset Status
    # -------------------------------------------------------------
    st.subheader("📌 Dataset Status")
    status_cols = st.columns(4)
    status_cols[0].metric("Source File", raw_filename)
    status_cols[1].metric("Original Rows", f"{len(raw_df):,}")
    status_cols[2].metric("Original Columns", f"{len(raw_df.columns):,}")
    active_dataset = st.session_state.get("active_dataset_name", "Original Dataset")
    status_cols[3].metric("Active for Analysis", active_dataset)

    # Scan for cleaning opportunities
    opps = detect_cleaning_opportunities(raw_df)

    # -------------------------------------------------------------
    # 2. Recommended Actions Reference
    # -------------------------------------------------------------
    with st.expander("💡 Recommended Cleaning Actions from Quality Assessment (Phase 6)", expanded=False):
        st.write("These recommendations were generated based on automated quality checks. "
                 "Review them below to help guide your cleaning decisions.")
        semantic_types_df = detect_semantic_types(raw_df)
        quality_report = analyze_data_quality(raw_df, semantic_types_df)
        recs_report = generate_cleaning_recommendations(raw_df, quality_report, semantic_types_df)
        recs_df = recs_report.get("recommendations_df", pd.DataFrame())
        if not recs_df.empty:
            display_recs = recs_df[["Priority", "Category", "Column", "Recommended Action", "Method", "Reason"]].copy()
            st.dataframe(display_recs, use_container_width=True)
        else:
            st.success("✅ No critical data hygiene issues detected by the recommendation engine.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. Interactive Cleaning Operations Configuration
    # -------------------------------------------------------------
    st.subheader("⚙️ Configure Cleaning Operations")
    st.write("Select the specific cleaning operations you want to apply. Unchecked operations will not be performed.")

    # Container 1: Structural & Duplicates
    with st.expander("1. Structural Cleaning & Duplicate Removal", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            has_empty_cols = len(opps["empty_columns"]) > 0
            empty_cols_label = f"Drop Entirely Empty Columns ({len(opps['empty_columns'])} detected)"
            do_drop_empty = st.checkbox(
                empty_cols_label,
                value=has_empty_cols,
                help="Drops columns where all values are null/NaN.",
            )
            if has_empty_cols:
                st.caption(f"Empty columns: `{', '.join(opps['empty_columns'])}`")

        with c2:
            has_duplicates = opps["duplicate_rows_count"] > 0
            dup_label = f"Remove Exact Duplicate Rows ({opps['duplicate_rows_count']:,} detected)"
            do_remove_dups = st.checkbox(
                dup_label,
                value=has_duplicates,
                help="Removes duplicate rows while keeping the first occurrence.",
            )
            if has_duplicates:
                st.caption(f"Preserves first occurrence (`keep='first'`).")

    # Container 2: Missing Value Handling
    with st.expander("2. Missing Value Treatment", expanded=True):
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.markdown("**Row-Level Missing Action**")
            do_drop_missing_rows = st.checkbox(
                "Drop rows with missing values",
                value=False,
                help="Removes records that contain missing values.",
            )
            drop_how = "any"
            missing_subset = None
            if do_drop_missing_rows:
                drop_how = st.radio(
                    "Drop condition:",
                    options=["any", "all"],
                    format_func=lambda x: "Drop if ANY selected column is missing" if x == "any" else "Drop only if ALL selected columns are missing",
                )
                missing_subset = st.multiselect(
                    "Apply row drop to specific columns (leave blank for all):",
                    options=list(raw_df.columns),
                    default=None,
                )

        impute_strategies: dict[str, dict[str, Any]] = {}
        with m_col2:
            st.markdown("**Column-Level Imputation Strategies**")
            missing_cols_dict = opps["missing_value_columns"]
            if not missing_cols_dict:
                st.success("✅ No missing values found in any column.")
            else:
                for col_name, col_info in missing_cols_dict.items():
                    if col_info["is_empty"] and do_drop_empty:
                        continue  # Will be dropped

                    is_num = col_info["is_numerical"]
                    default_strategies = (
                        ["Do Not Impute", "Median", "Mean", "Constant"]
                        if is_num
                        else ["Do Not Impute", "Mode (Most Frequent)", "Constant"]
                    )

                    col_ctrl1, col_ctrl2 = st.columns([2, 1])
                    with col_ctrl1:
                        selected_strat = st.selectbox(
                            f"'{col_name}' ({col_info['count']} missing, {col_info['percentage']}%):",
                            options=default_strategies,
                            key=f"impute_strat_{col_name}",
                        )

                    fill_val = None
                    with col_ctrl2:
                        if selected_strat == "Constant":
                            if is_num:
                                fill_val = st.number_input(f"Value for '{col_name}':", value=0.0, key=f"const_{col_name}")
                            else:
                                fill_val = st.text_input(f"Value for '{col_name}':", value="Missing", key=f"const_{col_name}")

                    if selected_strat != "Do Not Impute":
                        strat_key = selected_strat.split(" ")[0].lower()
                        impute_strategies[col_name] = {
                            "strategy": strat_key,
                            "fill_value": fill_val,
                        }

    # Container 3: Data-Type Conversions
    type_conversions: dict[str, str] = {}
    with st.expander("3. Safe Data-Type Conversions", expanded=False):
        st.write("Safely cast columns to appropriate analytical types. Values failing conversion are coerced to null with detailed warnings.")
        type_candidates = opps["type_conversion_candidates"]
        
        convertible_cols = [c for c in raw_df.columns if c not in opps["empty_columns"]]
        selected_conv_cols = st.multiselect(
            "Select columns to convert data type:",
            options=convertible_cols,
            default=list(type_candidates.keys()),
        )

        if selected_conv_cols:
            for c_col in selected_conv_cols:
                suggested = type_candidates.get(c_col, {}).get("suggested_type", "None")
                type_opts = ["Keep Current Type", "Numeric (Float)", "Nullable Integer (Int64)", "Datetime", "Boolean", "String"]
                default_idx = 0
                if suggested == "numeric":
                    default_idx = 1
                elif suggested == "datetime":
                    default_idx = 3

                t_sel = st.selectbox(
                    f"Target type for '{c_col}' (Current: {raw_df[c_col].dtype}):",
                    options=type_opts,
                    index=default_idx,
                    key=f"target_type_{c_col}",
                )
                if t_sel == "Numeric (Float)":
                    type_conversions[c_col] = "numeric"
                elif t_sel == "Nullable Integer (Int64)":
                    type_conversions[c_col] = "integer"
                elif t_sel == "Datetime":
                    type_conversions[c_col] = "datetime"
                elif t_sel == "Boolean":
                    type_conversions[c_col] = "boolean"
                elif t_sel == "String":
                    type_conversions[c_col] = "string"

    # Container 4: Text Standardization
    text_operations: dict[str, dict[str, Any]] = {}
    with st.expander("4. Text Standardization & Cleaning", expanded=False):
        st.write("Trim leading/trailing whitespace and standardize character casing for text fields.")
        text_cols = [
            c for c in raw_df.columns
            if (pd.api.types.is_object_dtype(raw_df[c]) or pd.api.types.is_string_dtype(raw_df[c]))
            and c not in opps["empty_columns"]
        ]

        if not text_cols:
            st.info("No text/string columns available for standardization.")
        else:
            selected_text_cols = st.multiselect(
                "Select text columns to standardize:",
                options=text_cols,
                default=list(opps["text_standardization_candidates"].keys()),
            )

            for t_col in selected_text_cols:
                tc1, tc2 = st.columns(2)
                with tc1:
                    do_trim = st.checkbox(f"Trim whitespace on '{t_col}'", value=True, key=f"trim_{t_col}")
                with tc2:
                    case_opt = st.selectbox(
                        f"Casing for '{t_col}':",
                        options=["None (Keep Original)", "lowercase", "UPPERCASE", "Title Case"],
                        index=0,
                        key=f"case_{t_col}",
                    )
                
                casing_map = {
                    "None (Keep Original)": "none",
                    "lowercase": "lower",
                    "UPPERCASE": "upper",
                    "Title Case": "title",
                }
                if do_trim or casing_map[case_opt] != "none":
                    text_operations[t_col] = {
                        "trim": do_trim,
                        "casing": casing_map[case_opt],
                    }

    # Container 5: Outlier Handling
    outlier_clipping: list[str] = []
    with st.expander("5. Numerical Outlier Handling (IQR Winsorization)", expanded=False):
        st.write("Clips extreme statistical outliers to IQR bounds: `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. "
                 "Rows are **not** dropped; extreme values are clamped to threshold limits.")
        outlier_cols_dict = opps["outlier_candidates"]

        if not outlier_cols_dict:
            st.success("✅ No extreme numerical outliers detected using 1.5× IQR threshold.")
        else:
            for o_col, o_info in outlier_cols_dict.items():
                do_clip = st.checkbox(
                    f"Clip outliers in '{o_col}' ({o_info['outlier_count']} outliers, bounds: [{o_info['lower_bound']}, {o_info['upper_bound']}])",
                    value=False,
                    key=f"clip_{o_col}",
                    help=f"Clamps values below {o_info['lower_bound']} and above {o_info['upper_bound']} to the threshold.",
                )
                if do_clip:
                    outlier_clipping.append(o_col)

    # -------------------------------------------------------------
    # Build Cleaning Pipeline Configuration
    # -------------------------------------------------------------
    cleaning_config: dict[str, Any] = {
        "drop_empty_columns": do_drop_empty,
        "empty_columns_to_drop": opps["empty_columns"] if do_drop_empty else None,
        "remove_duplicates": do_remove_dups,
        "duplicate_keep": "first",
        "drop_missing_rows": do_drop_missing_rows,
        "missing_rows_subset": missing_subset if (do_drop_missing_rows and missing_subset) else None,
        "missing_rows_how": drop_how,
        "impute_missing": impute_strategies,
        "convert_types": type_conversions,
        "standardize_text": text_operations,
        "clip_outliers": outlier_clipping,
    }

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. Preview & Apply Actions
    # -------------------------------------------------------------
    st.subheader("👁️ Preview Changes & Execute Cleaning")
    
    # Run a dry-run preview simulation
    preview_df, preview_logs, preview_summary = apply_cleaning_pipeline(raw_df, cleaning_config)

    st.write(
        f"**Proposed Operations:** {len(preview_logs)} operation(s) configured. "
        f"Estimated rows: **{preview_summary['before_rows']:,} → {preview_summary['after_rows']:,}** | "
        f"Estimated missing cells: **{preview_summary['before_missing']:,} → {preview_summary['after_missing']:,}**"
    )

    with st.expander("Show Preview of Transformed Sample Records (First 5 Rows)", expanded=False):
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.caption("Original Raw Sample:")
            st.dataframe(raw_df.head(5), use_container_width=True)
        with p_c2:
            st.caption("Cleaned Preview Sample:")
            st.dataframe(preview_df.head(5), use_container_width=True)

    # Apply Button
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        apply_clicked = st.button("✨ Apply Selected Cleaning", type="primary", use_container_width=True)

    if apply_clicked:
        # Execute pipeline starting from raw_df copy
        cleaned_df, applied_logs, applied_summary = apply_cleaning_pipeline(raw_df, cleaning_config)

        # Store in session state
        st.session_state["cleaned_data"] = cleaned_df
        st.session_state["cleaning_log"] = applied_logs
        st.session_state["cleaning_summary"] = applied_summary
        st.session_state["cleaned_filename"] = f"cleaned_{raw_filename}"
        
        # Default active dataset to Cleaned Dataset after cleaning
        st.session_state["data"] = cleaned_df
        st.session_state["filename"] = f"cleaned_{raw_filename}"
        st.session_state["active_dataset_name"] = "Cleaned Dataset"

        st.success("🎉 Data cleaning successfully executed and saved as an isolated cleaned dataset!")
        st.rerun()

    # -------------------------------------------------------------
    # 5. Results & Audit Trail Log
    # -------------------------------------------------------------
    if "cleaned_data" in st.session_state and st.session_state["cleaned_data"] is not None:
        st.markdown("---")
        st.subheader("📊 Cleaning Results & Audit Trail")

        c_df: pd.DataFrame = st.session_state["cleaned_data"]
        c_summary: dict[str, Any] = st.session_state.get("cleaning_summary", {})
        c_logs: list[dict[str, Any]] = st.session_state.get("cleaning_log", [])

        # Comparison metrics
        res_cols = st.columns(4)
        res_cols[0].metric(
            "Row Count",
            f"{c_summary.get('after_rows', len(c_df)):,}",
            delta=f"{c_summary.get('after_rows', len(c_df)) - c_summary.get('before_rows', len(raw_df)):,} rows",
            delta_color="inverse" if (c_summary.get('after_rows', len(c_df)) < c_summary.get('before_rows', len(raw_df))) else "normal",
        )
        res_cols[1].metric(
            "Column Count",
            f"{c_summary.get('after_cols', len(c_df.columns)):,}",
            delta=f"{c_summary.get('after_cols', len(c_df.columns)) - c_summary.get('before_cols', len(raw_df.columns)):,} cols",
        )
        res_cols[2].metric(
            "Missing Values",
            f"{c_summary.get('after_missing', int(c_df.isna().sum().sum())):,}",
            delta=f"{c_summary.get('after_missing', 0) - c_summary.get('before_missing', 0):,} cells",
            delta_color="inverse",
        )
        res_cols[3].metric(
            "Duplicate Rows",
            f"{c_summary.get('after_duplicates', int(c_df.duplicated().sum())):,}",
            delta=f"{c_summary.get('after_duplicates', 0) - c_summary.get('before_duplicates', 0):,} dups",
            delta_color="inverse",
        )

        # Cleaning Log Table
        st.markdown("### 📋 Cleaning Operations Log")
        if c_logs:
            log_rows = []
            for entry in c_logs:
                log_rows.append({
                    "Timestamp": entry.get("timestamp"),
                    "Operation": entry.get("operation"),
                    "Column(s)": ", ".join(entry.get("columns", [])) if entry.get("columns") else "Dataset-level",
                    "Strategy": entry.get("strategy"),
                    "Affected Count": f"{entry.get('affected_count', 0):,} {entry.get('affected_type', '')}",
                    "Details": entry.get("details"),
                    "Warnings": "; ".join(entry.get("warnings", [])) if entry.get("warnings") else "None",
                })
            log_table = pd.DataFrame(log_rows)
            st.dataframe(log_table, use_container_width=True)
        else:
            st.info("No cleaning operations were selected or applied.")

        # Cleaned dataset preview
        with st.expander("Preview Complete Cleaned Dataset (First 15 Rows)", expanded=False):
            st.dataframe(c_df.head(15), use_container_width=True)

        # -------------------------------------------------------------
        # 6. Export / Download
        # -------------------------------------------------------------
        st.markdown("### 📥 Download Cleaned Dataset")
        dl_c1, dl_c2 = st.columns(2)

        with dl_c1:
            csv_data = c_df.to_csv(index=False).encode("utf-8")
            base_name = raw_filename.rsplit(".", 1)[0]
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_data,
                file_name=f"{base_name}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl_c2:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    c_df.to_excel(writer, index=False, sheet_name="Cleaned Data")
                st.download_button(
                    label="📥 Download Cleaned Excel (XLSX)",
                    data=excel_buffer.getvalue(),
                    file_name=f"{base_name}_cleaned.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Excel export unavailable: {e}")

        # -------------------------------------------------------------
        # 7. Dataset Choice for Downstream Analysis
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Active Dataset for Downstream Analysis")
        st.write(
            "Choose which dataset will be used across downstream exploratory analysis pages "
            "(`Profile`, `Quality Report`, `Analysis`, `Visualizations`, `Insights`, `KPIs`)."
        )

        current_choice = st.session_state.get("active_dataset_name", "Cleaned Dataset")
        choice_idx = 1 if current_choice == "Cleaned Dataset" else 0

        chosen_dataset = st.radio(
            "Select active dataset:",
            options=["Original Dataset", "Cleaned Dataset"],
            index=choice_idx,
            horizontal=True,
            help="Switching here immediately updates the active dataset evaluated by all subsequent pages.",
        )

        if chosen_dataset != st.session_state.get("active_dataset_name"):
            if chosen_dataset == "Cleaned Dataset":
                st.session_state["data"] = c_df
                st.session_state["filename"] = f"cleaned_{raw_filename}"
                st.session_state["active_dataset_name"] = "Cleaned Dataset"
            else:
                st.session_state["data"] = raw_df
                st.session_state["filename"] = raw_filename
                st.session_state["active_dataset_name"] = "Original Dataset"
            st.success(f"✅ Active dataset updated to **{chosen_dataset}**!")
            st.rerun()

        st.info(
            f"ℹ️ Currently active: **{st.session_state.get('active_dataset_name')}** "
            f"({len(st.session_state['data']):,} rows, {len(st.session_state['data'].columns):,} columns)."
        )
