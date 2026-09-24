"""
utils.py

Small shared helpers: export functions (CSV / formatted Excel) and a
couple of generic utilities used by app.py.
"""

import io
from typing import Optional

import pandas as pd


EXPORT_COLUMNS = [
    "Rank", "Candidate", "Email", "Phone", "Skills", "Education",
    "Experience", "Matched Skills", "Missing Skills",
    "TF-IDF Similarity", "Skill Match", "Final Match Score",
]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return the results DataFrame as CSV bytes, ready for a Streamlit
    download_button."""
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    export_df = df[cols] if cols else df
    return export_df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Screening Results") -> bytes:
    """
    Return the results DataFrame as a formatted .xlsx file (bytes), ready
    for a Streamlit download_button. Formatting includes bold headers,
    readable column widths, and percentage-style number formatting on the
    score columns.
    """
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    export_df = df[cols] if cols else df

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]

        # Bold header row.
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)

        # Auto-size columns based on the longest value in each one.
        for col_idx, column in enumerate(export_df.columns, start=1):
            max_len = max(
                [len(str(column))] + [len(str(v)) for v in export_df[column].tolist()]
            )
            col_letter = worksheet.cell(row=1, column=col_idx).column_letter
            worksheet.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)

        # Percentage-style display for the three score columns (values are
        # stored as plain numbers like 91, so this appends a literal '%'
        # via a number format rather than converting to a 0-1 fraction).
        percent_cols = ["TF-IDF Similarity", "Skill Match", "Final Match Score"]
        for col_name in percent_cols:
            if col_name not in export_df.columns:
                continue
            col_idx = list(export_df.columns).index(col_name) + 1
            col_letter = worksheet.cell(row=1, column=col_idx).column_letter
            for row in range(2, len(export_df) + 2):
                worksheet[f"{col_letter}{row}"].number_format = '0"%"'

    buffer.seek(0)
    return buffer.getvalue()


def format_warning_list(warnings: list) -> Optional[str]:
    """Combine a list of per-file warning strings into one display-ready
    message, or return None if there are no warnings."""
    if not warnings:
        return None
    return "\n".join(f"- {w}" for w in warnings)
