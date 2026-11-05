"""Overview, gate statistics, and TMC summary export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from traffic_tracker.config import APP_NAME, INTERVAL_SECONDS
from traffic_tracker.core.geometry import Geometry, movement_name


VEHICLE_LABEL = "Number of all vehicles"
REPORT_CLASS_ROWS = (
    ("car", "Car count"),
    ("van", "Van count"),
    ("truck", "Truck count"),
    ("motorcycle", "Motorcycle count"),
    ("bus", "Bus count"),
)


@dataclass
class Trip:
    video_name: str
    track_id: int
    start_destination: str
    start_time_s: float
    end_destination: str
    end_time_s: float
    vehicle_class: str = "vehicle"
    path: list[tuple[float, float]] = field(default_factory=list)
    evidence: str = "observed"

    @property
    def route_name(self) -> str:
        return movement_name(self.start_destination, self.end_destination)

    @property
    def completed_at_s(self) -> float:
        return self.end_time_s


def interval_label(start_s: float, clock_start: datetime | None) -> tuple[str, str]:
    end_s = start_s + INTERVAL_SECONDS
    if clock_start is None:
        return _elapsed_label(start_s), _elapsed_label(end_s)
    start_dt = clock_start + timedelta(seconds=start_s)
    end_dt = clock_start + timedelta(seconds=end_s)
    return start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M")


def event_time_label(seconds: float, clock_start: datetime | None) -> str:
    if clock_start is None:
        return _elapsed_label(seconds)
    return (clock_start + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def period_label(start_s: float, clock_start: datetime | None) -> str:
    end_s = start_s + INTERVAL_SECONDS
    if clock_start is None:
        return f"{_hms(start_s)} - {_hms(end_s)}"
    start_dt = clock_start + timedelta(seconds=start_s)
    end_dt = clock_start + timedelta(seconds=end_s)
    return f"{start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}"


def _elapsed_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def tmc_tables_close(left: pd.DataFrame, right: pd.DataFrame, *, max_abs_diff: int = 1) -> bool:
    """True when route counts match within a small per-cell tolerance."""
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    count_cols = [c for c in left.columns if c not in ("Interval Start", "Interval End")]
    for col in count_cols:
        if (left[col].astype(int) - right[col].astype(int)).abs().max() > max_abs_diff:
            return False
    return True


def build_trip_table(trips: list[Trip], clock_start: datetime | None = None) -> pd.DataFrame:
    rows = [
        {
            "Start Destination": trip.start_destination,
            "Start Time": event_time_label(trip.start_time_s, clock_start),
            "End Destination": trip.end_destination,
            "End Time": event_time_label(trip.end_time_s, clock_start),
        }
        for trip in trips
    ]
    columns = ["Start Destination", "Start Time", "End Destination", "End Time"]
    return pd.DataFrame(rows, columns=columns)


def _bin_count(duration_s: float, trips: list[Trip]) -> int:
    if duration_s <= 0 and trips:
        duration_s = max(trip.end_time_s for trip in trips)
    if duration_s <= 0:
        duration_s = INTERVAL_SECONDS
    bin_count = max(1, int(duration_s // INTERVAL_SECONDS) + (1 if duration_s % INTERVAL_SECONDS else 0))
    if duration_s > 0 and duration_s % INTERVAL_SECONDS == 0:
        bin_count = max(1, int(duration_s // INTERVAL_SECONDS))
    return bin_count


def _trip_bin(trip: Trip, bin_count: int) -> int:
    idx = int(trip.end_time_s // INTERVAL_SECONDS)
    if idx < 0:
        return 0
    if idx >= bin_count:
        return bin_count - 1
    return idx


def build_tmc_table(
    trips: list[Trip],
    route_names: list[str],
    duration_s: float,
    clock_start: datetime | None = None,
) -> pd.DataFrame:
    """One row per 15-min bin, one column per movement name, plus Total."""
    bin_count = _bin_count(duration_s, trips)
    counts: list[dict[str, int]] = [dict.fromkeys(route_names, 0) for _ in range(bin_count)]
    for trip in trips:
        if trip.route_name not in route_names:
            continue
        counts[_trip_bin(trip, bin_count)][trip.route_name] += 1

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


def _class_count(trips: list[Trip], cls: str) -> int:
    return sum(1 for trip in trips if trip.vehicle_class == cls)


def build_overview_rows(trips: list[Trip]) -> list[tuple[str, int | str]]:
    rows: list[tuple[str, int | str]] = [("Overall statistics for all vehicles seen", "")]
    rows.extend((label, _class_count(trips, cls)) for cls, label in REPORT_CLASS_ROWS)
    rows.append((VEHICLE_LABEL, len(trips)))
    return rows


def build_overview_table(
    trips: list[Trip],
    *,
    video_name: str,
    duration_s: float,
    clock_start: datetime | None,
    device_label: str,
) -> pd.DataFrame:
    clock_text = clock_start.strftime("%A %B %d %Y") if clock_start else "Elapsed video time"
    rows = [
        ("Report", f"{APP_NAME} Report"),
        ("Video", video_name),
        ("Time basis", clock_text),
        ("Duration", _elapsed_label(duration_s)),
        ("Device", device_label),
        ("", ""),
        *[(label, value) for label, value in build_overview_rows(trips)],
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def build_gate_statistics_table(trips: list[Trip], geometry: Geometry) -> pd.DataFrame:
    origins = geometry.origin_names()
    bounds = geometry.bound_names()
    categories = [label for _cls, label in REPORT_CLASS_ROWS] + [VEHICLE_LABEL]
    data: dict[str, list[int]] = {"Category": categories}
    for origin in origins:
        data[f"{origin} (Entry)"] = [
            sum(1 for trip in trips if trip.start_destination == origin and trip.vehicle_class == cls)
            for cls, _label in REPORT_CLASS_ROWS
        ] + [sum(1 for trip in trips if trip.start_destination == origin)]
    for bound in bounds:
        data[f"{bound} (Exit)"] = [
            sum(1 for trip in trips if trip.end_destination == bound and trip.vehicle_class == cls)
            for cls, _label in REPORT_CLASS_ROWS
        ] + [sum(1 for trip in trips if trip.end_destination == bound)]
    return pd.DataFrame(data, columns=["Category", *[col for col in data if col != "Category"]])


def build_tmc_summary_table(
    trips: list[Trip],
    geometry: Geometry,
    duration_s: float,
    clock_start: datetime | None = None,
) -> pd.DataFrame:
    origins = geometry.origin_names()
    bounds = geometry.bound_names()
    bin_count = _bin_count(duration_s, trips)
    columns = ["Period"]
    for origin in origins:
        columns.extend([f"{origin}-{bound}" for bound in bounds])
        columns.append(f"{origin} Total")

    def empty_counts() -> dict[str, int]:
        return {col: 0 for col in columns if col != "Period"}

    bins = [empty_counts() for _ in range(bin_count)]
    for trip in trips:
        key = f"{trip.start_destination}-{trip.end_destination}"
        origin_total = f"{trip.start_destination} Total"
        idx = _trip_bin(trip, bin_count)
        if key in bins[idx]:
            bins[idx][key] += 1
        if origin_total in bins[idx]:
            bins[idx][origin_total] += 1

    rows: list[dict[str, object]] = []
    hour_size = max(1, 3600 // INTERVAL_SECONDS)
    hour_acc = empty_counts()
    sequence = empty_counts()
    bins_in_hour = 0

    def flush_hour() -> None:
        nonlocal hour_acc, bins_in_hour
        if bins_in_hour == 0:
            return
        rows.append({"Period": "Total (hourly)", **hour_acc})
        hour_acc = empty_counts()
        bins_in_hour = 0

    for i, counts in enumerate(bins):
        rows.append({"Period": period_label(i * INTERVAL_SECONDS, clock_start), **counts})
        for key, value in counts.items():
            hour_acc[key] += value
            sequence[key] += value
        bins_in_hour += 1
        if bins_in_hour >= hour_size:
            flush_hour()
    flush_hour()
    rows.append({"Period": "Total (whole sequence)", **sequence})
    return pd.DataFrame(rows, columns=columns)


def _border() -> Border:
    side = Side(style="thin", color="BFBFBF")
    return Border(left=side, right=side, top=side, bottom=side)


def _write_overview(
    ws,
    *,
    video_name: str,
    duration_s: float,
    clock_start: datetime | None,
    device_label: str,
    trips: list[Trip],
) -> None:
    title_font = Font(name="Calibri", bold=True, size=16, color="1F4E79")
    label_font = Font(name="Calibri", size=11)
    value_font = Font(name="Calibri", size=11, bold=True)
    section_font = Font(name="Calibri", bold=True, size=12, color="1F4E79")

    clock_text = clock_start.strftime("%A %B %d %Y") if clock_start else "Elapsed video time"
    ws["B2"] = f"{APP_NAME} Report"
    ws["B2"].font = title_font
    ws["B3"] = video_name
    ws["B3"].font = label_font
    ws["B4"] = clock_text
    ws["B4"].font = label_font
    ws["B5"] = f"Duration: {_elapsed_label(duration_s)}"
    ws["B5"].font = label_font
    ws["B6"] = f"Device: {device_label}"
    ws["B6"].font = label_font

    row = 8
    for label, value in build_overview_rows(trips):
        ws.cell(row, 2, label).font = section_font if value == "" else label_font
        if value != "":
            cell = ws.cell(row, 4, value)
            cell.font = value_font
            cell.alignment = Alignment(horizontal="center")
        row += 1
    ws.column_dimensions["B"].width = 42
    ws.column_dimensions["D"].width = 14


def _write_gate_statistics(ws, trips: list[Trip], geometry: Geometry) -> int:
    title_font = Font(name="Calibri", bold=True, size=14, color="1F4E79")
    header_font = Font(name="Calibri", bold=True, color="FFFFFF")
    label_font = Font(name="Calibri", size=11)
    body_font = Font(name="Calibri", size=11)
    total_font = Font(name="Calibri", size=11, bold=True)
    entry_fill = PatternFill("solid", fgColor="2E75B6")
    exit_fill = PatternFill("solid", fgColor="C45911")
    alt_fill = PatternFill("solid", fgColor="F2F2F2")
    total_fill = PatternFill("solid", fgColor="D6EAF8")
    thin = _border()

    table = build_gate_statistics_table(trips, geometry)
    ws["B2"] = "Survey report - Gate Statistics"
    ws["B2"].font = title_font

    headers = [str(col) for col in table.columns]
    start_col = 2
    header_row = 4
    for offset, header in enumerate(headers):
        cell = ws.cell(header_row, start_col + offset, header if header else "")
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin
        if "Exit" in header:
            cell.fill = exit_fill
        else:
            cell.fill = entry_fill

    for r_idx, row in table.iterrows():
        excel_row = header_row + 1 + int(r_idx)
        is_total = str(row.iloc[0]) == VEHICLE_LABEL
        for c_idx, value in enumerate(row):
            cell = ws.cell(excel_row, start_col + c_idx, value)
            cell.border = thin
            cell.alignment = Alignment(horizontal="center" if c_idx else "left")
            cell.font = total_font if is_total else (label_font if c_idx == 0 else body_font)
            if is_total:
                cell.fill = total_fill
            elif excel_row % 2 == 0:
                cell.fill = alt_fill

    last_data_row = header_row + len(table)
    for offset, header in enumerate(headers):
        width = 22 if header else 28
        ws.column_dimensions[get_column_letter(start_col + offset)].width = width
    return last_data_row


def _write_tmc_summary(
    ws,
    trips: list[Trip],
    geometry: Geometry,
    duration_s: float,
    clock_start: datetime | None,
) -> None:
    title_font = Font(name="Calibri", bold=True, size=14, color="1F4E79")
    group_font = Font(name="Calibri", bold=True, color="FFFFFF")
    header_font = Font(name="Calibri", bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=11)
    total_font = Font(name="Calibri", size=11, bold=True)
    group_fills = [
        PatternFill("solid", fgColor="1F4E79"),
        PatternFill("solid", fgColor="2E75B6"),
        PatternFill("solid", fgColor="548235"),
        PatternFill("solid", fgColor="C45911"),
    ]
    col_fill = PatternFill("solid", fgColor="5B9BD5")
    hour_fill = PatternFill("solid", fgColor="FFF2CC")
    seq_fill = PatternFill("solid", fgColor="D6EAF8")
    alt_fill = PatternFill("solid", fgColor="F2F2F2")
    thin = _border()

    origins = geometry.origin_names()
    bounds = geometry.bound_names()
    table = build_tmc_summary_table(trips, geometry, duration_s, clock_start)

    ws["A1"] = "TMC Summary"
    ws["A1"].font = title_font

    group_row = 3
    header_row = 4
    ws.cell(header_row, 1, "Period").font = header_font
    ws.cell(header_row, 1).fill = col_fill
    ws.cell(header_row, 1).border = thin

    col = 3
    for index, origin in enumerate(origins):
        width = max(1, len(bounds)) + 1
        start = col
        end = col + width - 1
        ws.merge_cells(start_row=group_row, start_column=start, end_row=group_row, end_column=end)
        group = ws.cell(group_row, start, f"ENTRY {origin}")
        group.font = group_font
        group.fill = group_fills[index % len(group_fills)]
        group.alignment = Alignment(horizontal="center")
        for offset, bound in enumerate(bounds):
            cell = ws.cell(header_row, start + offset, bound)
            cell.font = header_font
            cell.fill = col_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin
        total_cell = ws.cell(header_row, end, "Total")
        total_cell.font = header_font
        total_cell.fill = col_fill
        total_cell.alignment = Alignment(horizontal="center")
        total_cell.border = thin
        col = end + 2

    for r_offset, row in table.iterrows():
        excel_row = header_row + 1 + int(r_offset)
        period = str(row["Period"])
        is_hour = period == "Total (hourly)"
        is_seq = period == "Total (whole sequence)"
        period_cell = ws.cell(excel_row, 1, period)
        period_cell.font = total_font if is_hour or is_seq else body_font
        period_cell.border = thin
        if is_seq:
            period_cell.fill = seq_fill
        elif is_hour:
            period_cell.fill = hour_fill

        col = 3
        for origin in origins:
            values = [row[f"{origin}-{bound}"] for bound in bounds] + [row[f"{origin} Total"]]
            for offset, value in enumerate(values):
                cell = ws.cell(excel_row, col + offset, int(value))
                cell.font = total_font if is_hour or is_seq else body_font
                cell.alignment = Alignment(horizontal="center")
                cell.border = thin
                if is_seq:
                    cell.fill = seq_fill
                elif is_hour:
                    cell.fill = hour_fill
                elif excel_row % 2 == 0:
                    cell.fill = alt_fill
            col += len(values) + 1

    ws.column_dimensions["A"].width = 28
    used_col = 3
    for _origin in origins:
        for _ in range(len(bounds) + 1):
            ws.column_dimensions[get_column_letter(used_col)].width = 10
            used_col += 1
        used_col += 1
    ws.freeze_panes = "A5"


def _embed_path_image(ws, image, last_row: int, dest: Path) -> Path | None:
    if image is None:
        return None
    import cv2
    from openpyxl.drawing.image import Image as ExcelImage

    frame = image
    height, width = frame.shape[:2]
    max_side = 1400
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    png_path = dest.with_name(f"{dest.stem}_paths.png")
    cv2.imwrite(str(png_path), frame)
    picture = ExcelImage(str(png_path))
    max_w = 720
    if picture.width and picture.width > max_w:
        ratio = max_w / picture.width
        picture.width = max_w
        if picture.height:
            picture.height = int(picture.height * ratio)
    ws.add_image(picture, f"B{last_row + 2}")
    return png_path


def export_tmc_workbook(
    path: str | Path,
    trips: list[Trip],
    geometry: Geometry,
    *,
    video_name: str,
    duration_s: float,
    clock_start: datetime | None,
    device_label: str,
    trip_count: int | None = None,
    table: pd.DataFrame | None = None,
    path_image=None,
) -> Path:
    del table, trip_count
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from openpyxl import Workbook

    wb = Workbook()
    overview = wb.active
    overview.title = "Overview"
    _write_overview(
        overview,
        video_name=video_name,
        duration_s=duration_s,
        clock_start=clock_start,
        device_label=device_label,
        trips=trips,
    )

    gates = wb.create_sheet("Gate Statistics")
    last_row = _write_gate_statistics(gates, trips, geometry)
    _embed_path_image(gates, path_image, last_row, path)

    tmc = wb.create_sheet("TMC Summary")
    _write_tmc_summary(tmc, trips, geometry, duration_s, clock_start)

    wb.save(path)
    return path
