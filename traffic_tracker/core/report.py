"""Build a 15-minute turning-movement count spreadsheet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from traffic_tracker.config import INTERVAL_SECONDS


@dataclass
class Trip:
    video_name: str
    track_id: int
    route_name: str
    completed_at_s: float
    vehicle_class: str


def interval_label(start_s: float, clock_start: datetime | None) -> tuple[str, str]:
    end_s = start_s + INTERVAL_SECONDS
    if clock_start is None:
        return _elapsed_label(start_s), _elapsed_label(end_s)
    start_dt = clock_start + timedelta(seconds=start_s)
    end_dt = clock_start + timedelta(seconds=end_s)
    return start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M")


def _elapsed_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def tmc_tables_close(left: pd.DataFrame, right: pd.DataFrame, *, max_abs_diff: int = 1) -> bool:
    """True when route counts match within a small per-cell tolerance."""
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    count_cols = [c for c in left.columns if c not in ("Interval Start", "Interval End")]
    for col in count_cols:
        if (left[col].astype(int) - right[col].astype(int)).abs().max() > max_abs_diff:
            return False
    return True


def build_tmc_table(
    trips: list[Trip],
    route_names: list[str],
    duration_s: float,
    clock_start: datetime | None = None,
) -> pd.DataFrame:
    """One row per 15-min bin, one column per named route, plus Total."""
    if duration_s <= 0 and trips:
        duration_s = max(trip.completed_at_s for trip in trips)
    if duration_s <= 0:
        duration_s = INTERVAL_SECONDS

    bin_count = max(1, int(duration_s // INTERVAL_SECONDS) + (1 if duration_s % INTERVAL_SECONDS else 0))
    # A video that lands exactly on a boundary does not need an extra empty bin.
    if duration_s > 0 and duration_s % INTERVAL_SECONDS == 0:
        bin_count = max(1, int(duration_s // INTERVAL_SECONDS))

    counts: list[dict[str, int]] = [dict.fromkeys(route_names, 0) for _ in range(bin_count)]
    for trip in trips:
        if trip.route_name not in route_names:
            continue
        idx = int(trip.completed_at_s // INTERVAL_SECONDS)
        if idx < 0:
            idx = 0
        if idx >= bin_count:
            idx = bin_count - 1
        counts[idx][trip.route_name] += 1

    rows = []
    for i, route_counts in enumerate(counts):
        start_s = i * INTERVAL_SECONDS
        start_label, end_label = interval_label(start_s, clock_start)
        row = {
            "Interval Start": start_label,
            "Interval End": end_label,
            **route_counts,
            "Total": sum(route_counts.values()),
        }
        rows.append(row)

    columns = ["Interval Start", "Interval End", *route_names, "Total"]
    return pd.DataFrame(rows, columns=columns)


def export_tmc_workbook(
    path: str | Path,
    table: pd.DataFrame,
    *,
    video_name: str,
    duration_s: float,
    clock_start: datetime | None,
    device_label: str,
    trip_count: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "TMC"

    header_font = Font(name="Calibri", bold=True, size=14, color="FFFFFF")
    meta_font = Font(name="Calibri", size=11, color="333333")
    col_font = Font(name="Calibri", bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=11)
    header_fill = PatternFill("solid", fgColor="1F4E79")
    col_fill = PatternFill("solid", fgColor="2E75B6")
    total_fill = PatternFill("solid", fgColor="D6EAF8")
    alt_fill = PatternFill("solid", fgColor="F2F2F2")
    thin = Border(
        left=Side(style="thin", color="BFBFBF"),
        right=Side(style="thin", color="BFBFBF"),
        top=Side(style="thin", color="BFBFBF"),
        bottom=Side(style="thin", color="BFBFBF"),
    )

    clock_text = clock_start.strftime("%Y-%m-%d %H:%M") if clock_start else "Elapsed video time"
    duration_text = _elapsed_label(duration_s)

    ws.merge_cells("A1:D1")
    ws["A1"] = "Traffic Tracker — Turning Movement Count"
    ws["A1"].font = header_font
    ws["A1"].fill = header_fill
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 22

    meta_lines = [
        f"Video: {video_name}",
        f"Duration: {duration_text}",
        f"Time basis: {clock_text}",
        f"Interval: 15 minutes",
        f"Vehicles counted: {trip_count}",
        f"Device: {device_label}",
    ]
    for offset, line in enumerate(meta_lines, start=2):
        ws.merge_cells(start_row=offset, start_column=1, end_row=offset, end_column=4)
        cell = ws.cell(offset, 1, line)
        cell.font = meta_font

    start_row = 9
    for r_idx, row in enumerate(dataframe_to_rows(table, index=False, header=True), start=start_row):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(r_idx, c_idx, value)
            cell.font = col_font if r_idx == start_row else body_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin
            if r_idx == start_row:
                cell.fill = col_fill
            elif r_idx % 2 == 0:
                cell.fill = alt_fill
            if table.columns[c_idx - 1] == "Total" and r_idx > start_row:
                cell.fill = total_fill
                cell.font = Font(name="Calibri", size=11, bold=True)

    ws.auto_filter.ref = f"A{start_row}:{get_column_letter(table.shape[1])}{start_row + len(table)}"
    ws.freeze_panes = f"A{start_row + 1}"

    for col_idx, name in enumerate(table.columns, start=1):
        width = max(14, len(str(name)) + 4)
        if name in ("Interval Start", "Interval End"):
            width = 22
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    wb.save(path)
    return path
