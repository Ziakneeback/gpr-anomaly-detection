import csv
import array
import math
import random
import re
import struct
import statistics
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


SAMPLE_COUNT = 240
TRACE_COUNT = 96
MAX_PREVIEW_TRACES = 700
MAX_PREVIEW_SAMPLES = 900
MAX_PREVIEW_CELLS = MAX_PREVIEW_TRACES * MAX_PREVIEW_SAMPLES


@dataclass
class DetectionResult:
    trace_index: int
    sample_index: int
    score: float
    amplitude: float
    label: str
    trace_from: int = 0
    trace_to: int = 0
    sample_from: int = 0
    sample_to: int = 0
    area: int = 1
    confidence: float = 0.0
    reason: str = ""
    depth_m: float = 0.0
    latitude: float | None = None
    longitude: float | None = None
    coordinate_text: str = ""


@dataclass
class ImportResult:
    data: list
    source_type: str
    details: str
    gps_points: list | None = None


class GPRProcessor:
    """Algorithms for simple ground penetrating radar anomaly detection."""

    @staticmethod
    def generate_demo_data(trace_count=TRACE_COUNT, sample_count=SAMPLE_COUNT):
        data = []
        anomalies = [
            (26, 86, 1.9, "metallic object"),
            (58, 132, -1.7, "void / weak reflection"),
            (76, 178, 1.55, "wet soil / strong boundary"),
        ]

        for trace in range(trace_count):
            row = []
            for sample in range(sample_count):
                depth_decay = math.exp(-sample / 165)
                wave = (
                    0.38 * math.sin(sample * 0.13 + trace * 0.05)
                    + 0.18 * math.sin(sample * 0.045 - trace * 0.09)
                )
                noise = random.gauss(0, 0.055)
                value = depth_decay * wave + noise

                for center_trace, center_sample, strength, _ in anomalies:
                    dt = trace - center_trace
                    ds = sample - center_sample
                    blob = math.exp(-((dt * dt) / 42 + (ds * ds) / 190))
                    value += strength * blob

                row.append(value)
            data.append(row)
        return data

    @staticmethod
    def load_file(path):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = GPRProcessor.load_text_matrix(path)
            return ImportResult(data, "CSV", "текстовая матрица амплитуд", GPRProcessor.load_gps_for_path(path))
        if suffix in {".gpr", ".gpr2", ".txt", ".dat"}:
            return GPRProcessor.load_gpr(path)
        raise ValueError("Поддерживаются файлы .csv, .gpr, .gpr2, .txt, .dat")

    @staticmethod
    def load_gpr(path):
        path = Path(path)
        gps_points = GPRProcessor.load_gps_for_path(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 350:
            raise ValueError(
                "Файл слишком большой для прямой загрузки в настольный интерфейс. "
                "Разделите профиль на фрагменты или экспортируйте участок в CSV/GPR меньшего размера."
            )
        raw = path.read_bytes()
        if not raw:
            raise ValueError("Файл пустой")

        text_result = GPRProcessor.try_load_text_bytes(raw)
        if text_result:
            return ImportResult(text_result, path.suffix.upper(), "распознан как текстовая матрица", gps_points)

        known_result = GPRProcessor.try_load_known_gpr2(raw)
        if known_result:
            data, details = known_result
            data, preview_details = GPRProcessor.prepare_preview_matrix(data)
            if preview_details:
                details = f"{details}; {preview_details}"
            return ImportResult(data, path.suffix.upper(), details, gps_points)

        data, details = GPRProcessor.load_binary_gpr(raw)
        data, preview_details = GPRProcessor.prepare_preview_matrix(data)
        if preview_details:
            details = f"{details}; {preview_details}"
        return ImportResult(data, path.suffix.upper(), details, gps_points)

    @staticmethod
    def load_gps_for_path(path):
        path = Path(path)
        gps_path = path.with_suffix(".gps")
        if not gps_path.exists():
            return []

        points = []
        try:
            with open(gps_path, "r", encoding="utf-8-sig") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    lat = float(parts[2])
                    if parts[3].upper().startswith("S"):
                        lat = -lat
                    lon = float(parts[4])
                    if parts[5].upper().startswith("W"):
                        lon = -lon
                    alt = float(parts[6])
                    points.append({"lat": lat, "lon": lon, "alt": alt})
        except Exception:
            return []
        return points

    @staticmethod
    def try_load_text_bytes(raw):
        head = raw[:4096]
        printable = sum(32 <= byte <= 126 or byte in (9, 10, 13) for byte in head)
        if printable / max(1, len(head)) < 0.82:
            return None
        for encoding in ("utf-8-sig", "cp1251", "latin-1"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            rows = GPRProcessor.parse_numeric_lines(text)
            if rows:
                return rows
        return None

    @staticmethod
    def parse_numeric_lines(text):
        rows = []
        splitter = re.compile(r"[;,\s]+")
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            parts = [part for part in splitter.split(line) if part]
            if len(parts) < 4:
                continue
            try:
                row = [float(part.replace(",", ".")) for part in parts]
            except ValueError:
                continue
            rows.append(row)

        if len(rows) < 4:
            return None
        width = len(rows[0])
        rows = [row for row in rows if len(row) == width]
        if len(rows) < 4 or width < 4:
            return None
        rows, _details = GPRProcessor.prepare_preview_matrix(rows)
        return rows

    @staticmethod
    def try_load_known_gpr2(raw):
        if len(raw) < 160:
            return None
        magic = struct.unpack_from("<I", raw, 0)[0]
        if magic != 0xFEDCBA98:
            return None

        trace_count = struct.unpack_from("<I", raw, 24)[0]
        sample_count = struct.unpack_from("<I", raw, 28)[0]
        channel_count = struct.unpack_from("<I", raw, 32)[0]
        if not (8 <= trace_count <= 500000 and 16 <= sample_count <= 8192 and 1 <= channel_count <= 8):
            return None

        bytes_per_value = 2
        data_bytes = trace_count * sample_count * channel_count * bytes_per_value
        data_offset = len(raw) - data_bytes
        if data_offset < 0 or data_offset >= len(raw):
            return None

        values = array.array("h")
        values.frombytes(raw[data_offset : data_offset + data_bytes])
        if len(values) < trace_count * sample_count * channel_count:
            return None

        channel_scores = []
        step_trace = max(1, trace_count // 600)
        step_sample = max(1, sample_count // 200)
        for channel in range(channel_count):
            sample_values = []
            non_zero = 0
            for trace in range(0, trace_count, step_trace):
                for sample in range(0, sample_count, step_sample):
                    index = (trace * sample_count + sample) * channel_count + channel
                    value = values[index]
                    sample_values.append(value)
                    if value:
                        non_zero += 1
            if len(sample_values) < 2:
                continue
            stdev = statistics.pstdev(sample_values)
            channel_scores.append((stdev * (non_zero / len(sample_values)), channel))

        if not channel_scores:
            return None
        _score, selected_channel = max(channel_scores, key=lambda item: item[0])

        data = []
        for trace in range(trace_count):
            row = []
            base = trace * sample_count * channel_count
            for sample in range(sample_count):
                value = values[base + sample * channel_count + selected_channel]
                row.append(float(value) / 32768.0)
            data.append(row)

        details = (
            f"GPR2 parser: трасс={trace_count}, отсчетов={sample_count}, "
            f"каналов={channel_count}, выбран канал={selected_channel + 1}, "
            f"смещение данных={data_offset} байт"
        )
        return data, details

    @staticmethod
    def prepare_preview_matrix(data):
        if not data or not data[0]:
            return data, ""
        rows = len(data)
        cols = len(data[0])
        if rows * cols <= MAX_PREVIEW_CELLS and rows <= MAX_PREVIEW_TRACES and cols <= MAX_PREVIEW_SAMPLES:
            return data, ""

        trace_step = max(1, math.ceil(rows / MAX_PREVIEW_TRACES))
        sample_step = max(1, math.ceil(cols / MAX_PREVIEW_SAMPLES))
        reduced = []
        for trace in range(0, rows, trace_step):
            trace_block = data[trace : min(rows, trace + trace_step)]
            reduced_row = []
            for sample in range(0, cols, sample_step):
                total = 0.0
                count = 0
                sample_end = min(cols, sample + sample_step)
                for source_row in trace_block:
                    for value in source_row[sample:sample_end]:
                        total += value
                        count += 1
                reduced_row.append(total / max(1, count))
            reduced.append(reduced_row)

        details = (
            f"для стабильной работы интерфейса построен preview "
            f"{len(reduced)}x{len(reduced[0])} из исходных {rows}x{cols}"
        )
        return reduced, details

    @staticmethod
    def decode_numeric_array(payload, dtype, size):
        usable = len(payload) - (len(payload) % size)
        type_code = {"int16": "h", "uint16": "H", "float32": "f"}[dtype]
        values = array.array(type_code)
        values.frombytes(payload[:usable])
        if values.itemsize != size:
            raise ValueError("Неподдерживаемый размер бинарного типа")
        return values

    @staticmethod
    def scaled_values(values, dtype, scale, limit=None):
        if limit is None:
            source = values
        else:
            source = values[:limit]
        if dtype == "uint16":
            mean = sum(source) / max(1, len(source))
            return [(float(value) - mean) / scale for value in source]
        return [float(value) / scale for value in source]

    @staticmethod
    def load_binary_gpr(raw):
        candidates = []
        formats = [
            ("int16", 2, "<h", 32768.0),
            ("uint16", 2, "<H", 65535.0),
            ("float32", 4, "<f", 1.0),
        ]
        widths = [128, 160, 200, 240, 256, 320, 400, 512, 768, 1024, 1500, 2048]

        offsets = [0, 128, 256, 512, 1024, 2048, 4096, 8192]
        for dtype, size, _fmt, scale in formats:
            for offset in offsets:
                if offset >= len(raw):
                    continue
                payload = raw[offset:]
                usable = len(payload) - (len(payload) % size)
                if usable < size * 64:
                    continue
                try:
                    values = GPRProcessor.decode_numeric_array(payload[:usable], dtype, size)
                except (ValueError, OverflowError):
                    continue
                sample_values = GPRProcessor.scaled_values(values, dtype, scale, limit=50000)

                for width in widths:
                    trace_count = len(values) // width
                    if trace_count < 8:
                        continue
                    stdev = statistics.pstdev(sample_values)
                    if not math.isfinite(stdev) or stdev < 1e-9:
                        continue
                    preference = 1.0 / (1.0 + abs(width - 512) / 512)
                    score = trace_count * preference * min(stdev, 10)
                    candidates.append((score, dtype, size, scale, width, trace_count, offset))

        if not candidates:
            raise ValueError(
                "Не удалось распознать бинарный .gpr/.gpr2. Нужен пример файла или описание формата прибора."
            )

        _score, dtype, size, scale, width, trace_count, offset = max(candidates, key=lambda item: item[0])
        payload = raw[offset:]
        usable = len(payload) - (len(payload) % size)
        numeric_values = GPRProcessor.decode_numeric_array(payload[:usable], dtype, size)
        values = GPRProcessor.scaled_values(numeric_values, dtype, scale, limit=trace_count * width)
        data = [values[index * width : (index + 1) * width] for index in range(trace_count)]
        details = f"бинарный импорт: {dtype}, смещение={offset} байт, трасс={trace_count}, отсчетов={width}"
        return data, details

    @staticmethod
    def load_csv(path):
        return GPRProcessor.load_text_matrix(path)

    @staticmethod
    def load_text_matrix(path):
        data = []
        with open(path, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                try:
                    values = [float(cell.replace(",", ".")) for cell in row]
                except ValueError as exc:
                    raise ValueError(f"Некорректное число в строке {line_number}") from exc
                data.append(values)

        if not data:
            raise ValueError("CSV-файл пустой")

        width = len(data[0])
        if any(len(row) != width for row in data):
            raise ValueError("Все строки CSV должны иметь одинаковое количество отсчетов")
        data, _details = GPRProcessor.prepare_preview_matrix(data)
        return data

    @staticmethod
    def save_csv(path, data):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    @staticmethod
    def mean_center(data):
        sample_count = len(data[0])
        means = []
        for sample in range(sample_count):
            means.append(sum(row[sample] for row in data) / len(data))
        return [[value - means[i] for i, value in enumerate(row)] for row in data]

    @staticmethod
    def gain_compensation(data, gain=1.15):
        sample_count = len(data[0])
        corrected = []
        for row in data:
            corrected_row = []
            for sample, value in enumerate(row):
                factor = 1 + gain * (sample / max(1, sample_count - 1))
                corrected_row.append(value * factor)
            corrected.append(corrected_row)
        return corrected

    @staticmethod
    def moving_average(data, radius=2):
        smoothed = []
        for row in data:
            result_row = []
            for index in range(len(row)):
                left = max(0, index - radius)
                right = min(len(row), index + radius + 1)
                result_row.append(sum(row[left:right]) / (right - left))
            smoothed.append(result_row)
        return smoothed

    @staticmethod
    def dewow_filter(data, radius=18):
        filtered = []
        for row in data:
            result_row = []
            for index, value in enumerate(row):
                left = max(0, index - radius)
                right = min(len(row), index + radius + 1)
                local_mean = sum(row[left:right]) / (right - left)
                result_row.append(value - local_mean)
            filtered.append(result_row)
        return filtered

    @staticmethod
    def normalize_traces(data):
        normalized = []
        for row in data:
            median = statistics.median(row)
            mad = statistics.median([abs(value - median) for value in row]) or 1e-9
            normalized.append([(value - median) / (1.4826 * mad) for value in row])
        return normalized

    @staticmethod
    def lateral_smoothing(data, radius=1):
        if radius <= 0:
            return [row[:] for row in data]
        rows = len(data)
        cols = len(data[0])
        result = []
        for trace in range(rows):
            left = max(0, trace - radius)
            right = min(rows, trace + radius + 1)
            row = []
            for sample in range(cols):
                row.append(sum(data[t][sample] for t in range(left, right)) / (right - left))
            result.append(row)
        return result

    @staticmethod
    def preprocess(data, remove_background=True, gain=True, smoothing=True):
        processed = [row[:] for row in data]
        processed = GPRProcessor.dewow_filter(processed)
        if remove_background:
            processed = GPRProcessor.mean_center(processed)
        if gain:
            processed = GPRProcessor.gain_compensation(processed)
        if smoothing:
            processed = GPRProcessor.moving_average(processed)
            processed = GPRProcessor.lateral_smoothing(processed)
        processed = GPRProcessor.normalize_traces(processed)
        return processed

    @staticmethod
    def detect_anomalies(data, threshold=3.0, min_distance=7, max_results=25):
        return GPRProcessor.detect_anomaly_zones(data, threshold, min_distance, max_results)

    @staticmethod
    def detect_anomaly_zones(data, threshold=3.0, min_distance=7, max_results=25):
        values = [abs(value) for row in data for value in row]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations) or 1e-9
        rows = len(data)
        cols = len(data[0])
        score_map = []
        mask = []
        for row in data:
            score_row = []
            mask_row = []
            for value in row:
                score = 0.6745 * (abs(value) - median) / mad
                score_row.append(score)
                mask_row.append(score >= threshold)
            score_map.append(score_row)
            mask.append(mask_row)

        visited = [[False] * cols for _ in range(rows)]
        zones = []
        for trace in range(rows):
            for sample in range(cols):
                if not mask[trace][sample] or visited[trace][sample]:
                    continue
                component = GPRProcessor.collect_component(mask, visited, trace, sample)
                if len(component) < 4:
                    continue
                zone = GPRProcessor.component_to_zone(component, data, score_map)
                if zone.score >= threshold and GPRProcessor.is_valid_zone(zone, rows, cols):
                    zones.append(zone)

        zones.sort(key=lambda item: (item.confidence, item.score, item.area), reverse=True)
        selected = []
        for zone in zones:
            too_close = any(
                abs(zone.trace_index - prev.trace_index) <= min_distance
                and abs(zone.sample_index - prev.sample_index) <= min_distance
                for prev in selected
            )
            if not too_close:
                selected.append(zone)
            if len(selected) >= max_results:
                break
        return selected

    @staticmethod
    def is_valid_zone(zone, rows, cols):
        top_margin = max(3, int(cols * 0.04))
        bottom_margin = max(3, int(cols * 0.035))
        if zone.sample_from < top_margin:
            return False
        if zone.sample_to >= cols - bottom_margin:
            return False

        width = zone.trace_to - zone.trace_from + 1
        height = zone.sample_to - zone.sample_from + 1
        if zone.area < 5:
            return False
        if width == 1 and height < 6:
            return False
        if height == 1 and width < 6:
            return False
        return True

    @staticmethod
    def collect_component(mask, visited, trace, sample):
        rows = len(mask)
        cols = len(mask[0])
        stack = [(trace, sample)]
        visited[trace][sample] = True
        component = []
        while stack:
            current_trace, current_sample = stack.pop()
            component.append((current_trace, current_sample))
            for dt, ds in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                next_trace = current_trace + dt
                next_sample = current_sample + ds
                if 0 <= next_trace < rows and 0 <= next_sample < cols:
                    if mask[next_trace][next_sample] and not visited[next_trace][next_sample]:
                        visited[next_trace][next_sample] = True
                        stack.append((next_trace, next_sample))
        return component

    @staticmethod
    def component_to_zone(component, data, score_map):
        trace_values = [trace for trace, _sample in component]
        sample_values = [sample for _trace, sample in component]
        trace_from = min(trace_values)
        trace_to = max(trace_values)
        sample_from = min(sample_values)
        sample_to = max(sample_values)
        peak_trace, peak_sample = max(component, key=lambda point: score_map[point[0]][point[1]])
        peak_score = score_map[peak_trace][peak_sample]
        peak_amplitude = data[peak_trace][peak_sample]
        area = len(component)

        min_trace_width = 9
        min_sample_height = 7
        if trace_to - trace_from + 1 < min_trace_width:
            extra = min_trace_width - (trace_to - trace_from + 1)
            trace_from = max(0, trace_from - extra // 2)
            trace_to = min(len(data) - 1, trace_to + math.ceil(extra / 2))
        if sample_to - sample_from + 1 < min_sample_height:
            extra = min_sample_height - (sample_to - sample_from + 1)
            sample_from = max(0, sample_from - extra // 2)
            sample_to = min(len(data[0]) - 1, sample_to + math.ceil(extra / 2))

        width = trace_to - trace_from + 1
        height = sample_to - sample_from + 1
        density = area / max(1, width * height)
        vertical_position = (sample_from + sample_to) / 2 / max(1, len(data[0]) - 1)

        confidence = min(0.98, 0.35 + peak_score / 18 + min(area, 80) / 180 + density * 0.2)
        if vertical_position < 0.08:
            confidence *= 0.75

        if peak_amplitude > 0:
            label = "перспективная зона: сильное отражение"
        else:
            label = "перспективная зона: ослабление/пустота"

        reasons = []
        reasons.append(f"пик score={peak_score:.2f}")
        reasons.append(f"размер зоны {width}x{height}")
        if density >= 0.35:
            reasons.append("связная компактная область")
        if vertical_position >= 0.08:
            reasons.append("расположена ниже поверхностных помех")
        if peak_amplitude < 0:
            reasons.append("есть признак ослабления сигнала, возможна полость или нарушение слоя")
        else:
            reasons.append("есть сильный контраст отражения, возможна граница объекта или каменная/металлическая структура")

        return DetectionResult(
            peak_trace,
            peak_sample,
            peak_score,
            peak_amplitude,
            label,
            trace_from,
            trace_to,
            sample_from,
            sample_to,
            area,
            confidence,
            "; ".join(reasons),
        )


class GPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Определение аномалий георадиолокационных сигналов")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.raw_data = GPRProcessor.generate_demo_data()
        self.processed_data = []
        self.results = []
        self.source_info = "демонстрационный набор"
        self.gps_points = []
        self.cell_w = 1
        self.cell_h = 1
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = None
        self.drag_moved = False
        self.detection_job = None
        self.draw_job = None
        self.manual_analysis = False
        self.last_no_results_alert = None
        self.busy = False

        self.threshold_var = tk.DoubleVar(value=3.0)
        self.antenna_var = tk.StringVar(value="400 МГц (1-3 м)")
        self.bg_var = tk.BooleanVar(value=True)
        self.gain_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Загружен демонстрационный набор данных")
        self.current_file_var = tk.StringVar(value="Файл: демонстрационные данные")

        self._build_ui()
        self.run_detection()

    def _build_ui(self):
        bg = "#e9edf2"
        panel_bg = "#17212b"
        panel_fg = "#e9f1f7"
        muted_fg = "#a9b7c6"
        accent = "#1fb6a6"
        accent_dark = "#159184"
        section_bg = "#f6f8fb"
        border = "#9aa8b6"

        self.configure(bg=bg)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"), background=bg, foreground="#14212d")
        style.configure("SideTitle.TLabel", font=("Segoe UI", 21, "bold"), background=panel_bg, foreground=panel_fg)
        style.configure("SideBody.TLabel", font=("Segoe UI", 10), background=panel_bg, foreground=muted_fg)
        style.configure("SideLabel.TLabel", font=("Segoe UI", 10, "bold"), background=panel_bg, foreground=panel_fg)
        style.configure("Status.TLabel", background="#22303d", foreground=panel_fg, padding=10)
        style.configure("Action.TButton", padding=(10, 8), font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", padding=(10, 8), font=("Segoe UI", 10, "bold"), background=accent, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", accent_dark), ("pressed", accent_dark)])
        style.configure("Panel.TCheckbutton", background=panel_bg, foreground=panel_fg, font=("Segoe UI", 10))
        style.map("Panel.TCheckbutton", background=[("active", panel_bg)], foreground=[("active", "#ffffff")])
        style.configure("TScale", background=panel_bg, troughcolor="#344454")
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground="#17212b", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#dce5ed", foreground="#17212b", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", accent)], foreground=[("selected", "#ffffff")])

        root = tk.Frame(self, bg=bg, padx=14, pady=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        panel = tk.Frame(root, width=300, bg=panel_bg, padx=18, pady=18)
        panel.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        panel.grid_propagate(False)

        ttk.Label(panel, text="GPR аномалии", style="SideTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(
            panel,
            text=(
                "Алгоритм ищет участки, где амплитуда отраженного сигнала "
                "резко отличается от фонового уровня."
            ),
            wraplength=260,
            style="SideBody.TLabel",
        ).pack(anchor="w", pady=(0, 18))

        ttk.Button(
            panel, text="Загрузить GPR / CSV", style="Accent.TButton", command=self.load_file
        ).pack(fill="x", pady=3)
        ttk.Button(
            panel,
            text="Создать демо-сигналы",
            style="Action.TButton",
            command=self.generate_demo,
        ).pack(fill="x", pady=3)
        ttk.Button(
            panel, text="Сохранить демо CSV", style="Action.TButton", command=self.save_demo
        ).pack(fill="x", pady=3)
        ttk.Button(
            panel, text="Сбросить масштаб карты", style="Action.TButton", command=self.reset_view
        ).pack(fill="x", pady=3)

        ttk.Separator(panel).pack(fill="x", pady=14)

        ttk.Label(panel, text="Порог аномальности", style="SideLabel.TLabel").pack(anchor="w")
        threshold = ttk.Scale(
            panel,
            from_=1.5,
            to=6.0,
            variable=self.threshold_var,
            command=self.schedule_detection,
        )
        threshold.pack(fill="x", pady=(4, 0))
        self.threshold_label = ttk.Label(panel, style="SideBody.TLabel")
        self.threshold_label.pack(anchor="w", pady=(0, 12))

        ttk.Label(panel, text="Антенна / глубина", style="SideLabel.TLabel").pack(anchor="w")
        antenna_box = ttk.Combobox(
            panel,
            textvariable=self.antenna_var,
            values=("1000 МГц (до 1 м)", "400 МГц (1-3 м)"),
            state="readonly",
            height=2,
        )
        antenna_box.pack(fill="x", pady=(4, 12))
        antenna_box.bind("<<ComboboxSelected>>", self.schedule_detection)

        ttk.Checkbutton(
            panel,
            text="Удаление фоновой составляющей",
            variable=self.bg_var,
            command=self.schedule_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            panel,
            text="Компенсация затухания с глубиной",
            variable=self.gain_var,
            command=self.schedule_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            panel,
            text="Сглаживание шума",
            variable=self.smooth_var,
            command=self.schedule_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Button(
            panel, text="Выполнить анализ", style="Accent.TButton", command=self.run_manual_detection
        ).pack(fill="x", pady=3)
        ttk.Button(
            panel, text="Сохранить отчет", style="Action.TButton", command=self.save_report
        ).pack(fill="x", pady=3)

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Label(panel, text="Загруженный файл", style="SideLabel.TLabel").pack(anchor="w")
        ttk.Label(panel, textvariable=self.current_file_var, wraplength=250, style="SideBody.TLabel").pack(
            fill="x", anchor="w", pady=(4, 8)
        )
        self.progress_bar = ttk.Progressbar(panel, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(0, 10))
        ttk.Label(panel, textvariable=self.status_var, wraplength=250, style="Status.TLabel").pack(fill="x", anchor="w")

        workspace = tk.Frame(root, bg=bg)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.rowconfigure(1, weight=1)
        workspace.columnconfigure(0, weight=3)
        workspace.columnconfigure(1, weight=2)

        ttk.Label(
            workspace,
            text="Радарограмма и найденные аномальные зоны",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.canvas = tk.Canvas(workspace, bg=section_bg, highlightthickness=2, highlightbackground=border)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_heatmap)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_heatmap)
        self.canvas.bind("<Button-4>", self.zoom_heatmap)
        self.canvas.bind("<Button-5>", self.zoom_heatmap)
        self.canvas.bind("<Configure>", self.schedule_draw)

        right = tk.Frame(workspace, bg=bg)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Список аномалий", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        columns = ("trace", "sample", "depth", "score", "confidence", "area", "coords", "label")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=16)
        headings = {
            "trace": "Трасса",
            "sample": "Отсчет",
            "depth": "Глубина",
            "score": "Оценка",
            "confidence": "Увер.",
            "area": "Зона",
            "coords": "Координаты",
            "label": "Тип",
        }
        widths = {
            "trace": 58,
            "sample": 60,
            "depth": 72,
            "score": 64,
            "confidence": 64,
            "area": 62,
            "coords": 150,
            "label": 180,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        ttk.Label(right, text="A-скан выбранной трассы", style="Title.TLabel").grid(
            row=2, column=0, sticky="w", pady=(16, 8)
        )
        self.trace_canvas = tk.Canvas(
            right, height=210, bg="#fbfcfe", highlightthickness=2, highlightbackground=border
        )
        self.trace_canvas.grid(row=3, column=0, sticky="ew")

    def generate_demo(self):
        if self.busy:
            return
        self.raw_data = GPRProcessor.generate_demo_data()
        self.source_info = "демонстрационный набор"
        self.gps_points = []
        self.current_file_var.set("Файл: демонстрационные данные")
        self.reset_view(redraw=False)
        self.status_var.set("Создан новый демонстрационный набор георадиолокационных сигналов")
        self.run_detection(show_alerts=True)
        self.show_success("Демо-данные готовы", "Демонстрационный набор создан и проанализирован.")

    def load_file(self):
        if self.busy:
            messagebox.showinfo("Операция выполняется", "Дождитесь завершения текущей загрузки или анализа.")
            return
        path = filedialog.askopenfilename(
            title="Выберите GPR/GPR2/CSV файл",
            filetypes=[
                ("GPR files", "*.gpr *.gpr2"),
                ("CSV files", "*.csv"),
                ("Text/Data files", "*.txt *.dat"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.start_busy(f"Загрузка файла: {Path(path).name}")
        threading.Thread(target=self.load_file_worker, args=(path,), daemon=True).start()

    def load_file_worker(self, path):
        try:
            imported = GPRProcessor.load_file(path)
            self.validate_imported_data(imported.data)
        except Exception as exc:
            self.after(0, lambda exc=exc: self.finish_load_error(exc))
            return
        self.after(0, lambda: self.finish_load_success(path, imported))

    def finish_load_error(self, exc):
        self.stop_busy()
        self.show_load_error(exc)

    def finish_load_success(self, path, imported):
        self.raw_data = imported.data
        self.source_info = f"{Path(path).name}: {imported.details}"
        self.gps_points = imported.gps_points or []
        self.current_file_var.set(f"Файл: {Path(path).name}")
        self.reset_view(redraw=False)
        self.status_var.set(f"Загружен файл: {self.source_info}")
        self.stop_busy()
        self.show_success(
            "Файл загружен",
            f"Файл {Path(path).name} успешно загружен.\n"
            f"Трасс: {len(self.raw_data)}, отсчетов: {len(self.raw_data[0])}, GPS-точек: {len(self.gps_points)}."
        )
        self.run_detection_async(show_alerts=True, success_alert=True)

    def start_busy(self, text):
        self.busy = True
        self.status_var.set(text)
        self.configure(cursor="watch")
        if hasattr(self, "progress_bar"):
            self.progress_bar.start(12)
        self.update_idletasks()

    def stop_busy(self):
        self.busy = False
        self.configure(cursor="")
        if hasattr(self, "progress_bar"):
            self.progress_bar.stop()

    def show_success(self, title, message):
        messagebox.showinfo(title, message)

    def validate_imported_data(self, data):
        if not data:
            raise ValueError("Файл не содержит числовых данных.")
        if not data[0]:
            raise ValueError("Трассы не содержат отсчетов.")
        width = len(data[0])
        if width < 8 or len(data) < 4:
            raise ValueError("Слишком мало трасс или отсчетов для построения B-scan.")
        if any(len(row) != width for row in data):
            raise ValueError("Матрица данных повреждена: строки имеют разную длину.")
        checked = 0
        for row in data[: min(len(data), 50)]:
            for value in row[: min(width, 50)]:
                checked += 1
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Файл содержит некорректные числовые значения.")
        if checked == 0:
            raise ValueError("Не удалось найти числовые значения для анализа.")

    def show_load_error(self, exc):
        message = str(exc)
        lower = message.lower()
        if "поддерживаются" in lower:
            advice = "Выберите файл .gpr, .gpr2, .csv, .txt или .dat."
        elif "распознать" in lower or "формат" in lower:
            advice = "Экспортируйте профиль в CSV или пришлите пример формата прибора для добавления точного парсера."
        elif "слишком большой" in lower:
            advice = "Разделите профиль на меньшие фрагменты или экспортируйте нужный участок."
        elif "пуст" in lower:
            advice = "Проверьте, что выбран правильный файл и он не поврежден."
        else:
            advice = "Проверьте файл, его расширение и наличие данных."
        messagebox.showerror("Ошибка загрузки файла", f"{message}\n\nЧто сделать: {advice}")
        self.status_var.set("Файл не загружен: проверьте формат или структуру данных")

    def show_analysis_error(self, exc):
        messagebox.showerror(
            "Ошибка анализа",
            "Файл был загружен, но анализ не выполнен.\n\n"
            f"Причина: {exc}\n\n"
            "Что сделать: попробуйте другой режим фильтрации, проверьте формат файла или экспортируйте профиль в CSV."
        )
        self.status_var.set("Анализ не выполнен: проверьте параметры и структуру файла")

    def reset_view(self, redraw=True):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if redraw:
            self.draw_heatmap()

    def clamp_view(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self.processed_data:
            data_width = len(self.processed_data) * self.cell_w
            data_height = len(self.processed_data[0]) * self.cell_h
        else:
            data_width = width * self.zoom
            data_height = height * self.zoom

        if data_width <= width:
            self.pan_x = (width - data_width) / 2
        else:
            self.pan_x = min(0, max(width - data_width, self.pan_x))

        if data_height <= height:
            self.pan_y = (height - data_height) / 2
        else:
            self.pan_y = min(0, max(height - data_height, self.pan_y))

    def save_demo(self):
        path = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        GPRProcessor.save_csv(path, self.raw_data)
        self.status_var.set(f"CSV сохранен: {Path(path).name}")

    def schedule_detection(self, _value=None):
        if hasattr(self, "threshold_label"):
            self.threshold_label.config(text=f"Текущее значение: {self.threshold_var.get():.2f}")
        if self.detection_job is not None:
            self.after_cancel(self.detection_job)
        self.detection_job = self.after(260, self.run_scheduled_detection)

    def run_scheduled_detection(self):
        self.detection_job = None
        self.run_detection_async(show_alerts=False, success_alert=False)

    def run_manual_detection(self):
        self.run_detection_async(show_alerts=True, success_alert=True)

    def run_detection_async(self, show_alerts=False, success_alert=False):
        if self.busy:
            if show_alerts:
                messagebox.showinfo("Операция выполняется", "Дождитесь завершения текущей операции.")
            return
        if not self.raw_data:
            if show_alerts:
                messagebox.showwarning(
                    "Нет данных",
                    "Сначала загрузите GPR/GPR2/CSV файл или создайте демонстрационные сигналы."
                )
            return
        self.start_busy("Выполняется анализ данных...")
        settings = {
            "remove_background": self.bg_var.get(),
            "gain": self.gain_var.get(),
            "smoothing": self.smooth_var.get(),
            "threshold": self.threshold_var.get(),
        }
        threading.Thread(
            target=self.detection_worker,
            args=(settings, show_alerts, success_alert),
            daemon=True,
        ).start()

    def detection_worker(self, settings, show_alerts, success_alert):
        try:
            self.validate_imported_data(self.raw_data)
            processed = GPRProcessor.preprocess(
                self.raw_data,
                remove_background=settings["remove_background"],
                gain=settings["gain"],
                smoothing=settings["smoothing"],
            )
            results = GPRProcessor.detect_anomalies(
                processed, threshold=settings["threshold"]
            )
        except Exception as exc:
            self.after(0, lambda exc=exc: self.finish_detection_error(exc, show_alerts))
            return
        self.after(0, lambda: self.finish_detection_success(processed, results, show_alerts, success_alert))

    def finish_detection_error(self, exc, show_alerts):
        self.stop_busy()
        if show_alerts:
            self.show_analysis_error(exc)
        else:
            self.status_var.set(f"Ошибка анализа: {exc}")

    def finish_detection_success(self, processed, results, show_alerts, success_alert):
        self.processed_data = processed
        self.results = results
        self.annotate_results()
        self.refresh_table()
        self.draw_heatmap()
        self.draw_trace(self.results[0].trace_index if self.results else 0)
        self.status_var.set(
            f"{self.source_info}\n"
            f"Трасс: {len(self.raw_data)}, отсчетов: {len(self.raw_data[0])}, "
            f"зон интереса: {len(self.results)}, GPS точек: {len(self.gps_points)}"
        )
        self.stop_busy()
        if not self.results:
            self.show_no_anomalies_alert(show_alerts)
            return
        if success_alert:
            self.show_success(
                "Анализ завершен",
                f"Анализ выполнен успешно.\nНайдено зон интереса: {len(self.results)}."
            )

    def schedule_draw(self, _event=None):
        if self.draw_job is not None:
            self.after_cancel(self.draw_job)
        self.draw_job = self.after(80, self.run_scheduled_draw)

    def run_scheduled_draw(self):
        self.draw_job = None
        self.draw_heatmap()

    def run_detection(self, show_alerts=False):
        if not self.raw_data:
            if show_alerts:
                messagebox.showwarning(
                    "Нет данных",
                    "Сначала загрузите GPR/GPR2/CSV файл или создайте демонстрационные сигналы."
                )
            return
        if self.detection_job is not None:
            self.after_cancel(self.detection_job)
            self.detection_job = None
        try:
            self.validate_imported_data(self.raw_data)
            self.threshold_label.config(text=f"Текущее значение: {self.threshold_var.get():.2f}")
            self.processed_data = GPRProcessor.preprocess(
                self.raw_data,
                remove_background=self.bg_var.get(),
                gain=self.gain_var.get(),
                smoothing=self.smooth_var.get(),
            )
            self.results = GPRProcessor.detect_anomalies(
                self.processed_data, threshold=self.threshold_var.get()
            )
        except Exception as exc:
            if show_alerts:
                self.show_analysis_error(exc)
            else:
                self.status_var.set(f"Ошибка анализа: {exc}")
            return
        self.annotate_results()
        self.refresh_table()
        self.draw_heatmap()
        self.draw_trace(self.results[0].trace_index if self.results else 0)
        self.status_var.set(
            f"{self.source_info}\n"
            f"Трасс: {len(self.raw_data)}, отсчетов: {len(self.raw_data[0])}, "
            f"зон интереса: {len(self.results)}, GPS точек: {len(self.gps_points)}"
        )
        if not self.results:
            self.show_no_anomalies_alert(show_alerts)

    def show_no_anomalies_alert(self, show_alerts):
        signature = (
            self.source_info,
            round(self.threshold_var.get(), 2),
            self.bg_var.get(),
            self.gain_var.get(),
            self.smooth_var.get(),
            self.antenna_var.get(),
        )
        self.status_var.set(
            f"{self.source_info}\n"
            f"Аномальные зоны не найдены. Попробуйте снизить порог или изменить фильтры."
        )
        if show_alerts and self.last_no_results_alert != signature:
            self.last_no_results_alert = signature
            messagebox.showinfo(
                "Аномалии не найдены",
                "При текущих настройках перспективные зоны не обнаружены.\n\n"
                "Что можно попробовать:\n"
                "1. Снизить порог аномальности.\n"
                "2. Включить удаление фона и сглаживание.\n"
                "3. Проверить, правильно ли выбрана антенна.\n"
                "4. Убедиться, что файл содержит профиль с полезным сигналом."
            )

    def max_depth_m(self):
        if self.antenna_var.get().startswith("1000"):
            return 1.0
        return 3.0

    def annotate_results(self):
        if not self.results or not self.processed_data:
            return
        sample_count = len(self.processed_data[0])
        max_depth = self.max_depth_m()
        for item in self.results:
            item.depth_m = item.sample_index / max(1, sample_count - 1) * max_depth
            gps = self.gps_for_trace(item.trace_index)
            if gps:
                item.latitude = gps["lat"]
                item.longitude = gps["lon"]
                item.coordinate_text = f"{gps['lat']:.6f}, {gps['lon']:.6f}"
            else:
                item.latitude = None
                item.longitude = None
                item.coordinate_text = f"трасса {item.trace_index}, отсчет {item.sample_index}"

    def gps_for_trace(self, trace_index):
        if not self.gps_points or not self.processed_data:
            return None
        if len(self.processed_data) <= 1:
            gps_index = 0
        else:
            gps_index = round(trace_index / (len(self.processed_data) - 1) * (len(self.gps_points) - 1))
        gps_index = max(0, min(len(self.gps_points) - 1, gps_index))
        return self.gps_points[gps_index]

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for index, item in enumerate(self.results, start=1):
            self.tree.insert(
                "",
                "end",
                iid=str(index - 1),
                values=(
                    item.trace_index,
                    item.sample_index,
                    f"{item.depth_m:.2f} м",
                    f"{item.score:.2f}",
                    f"{item.confidence * 100:.0f}%",
                    f"{item.trace_to - item.trace_from + 1}x{item.sample_to - item.sample_from + 1}",
                    item.coordinate_text,
                    item.label,
                ),
            )

    def draw_heatmap(self):
        if not self.processed_data:
            return
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        rows = len(self.processed_data)
        cols = len(self.processed_data[0])
        self.cell_w = (width / rows) * self.zoom
        self.cell_h = (height / cols) * self.zoom
        self.clamp_view()
        max_abs = self.display_scale(self.processed_data)

        target_traces = min(rows, int(180 + 35 * self.zoom))
        target_samples = min(cols, int(145 + 20 * self.zoom))
        step_trace = max(1, math.ceil(rows / max(1, target_traces)))
        step_sample = max(1, math.ceil(cols / max(1, target_samples)))
        for trace in range(0, rows, step_trace):
            for sample in range(0, cols, step_sample):
                value = self.processed_data[trace][sample] / max_abs
                color = self.value_to_color(value)
                x1 = self.pan_x + trace * self.cell_w
                y1 = self.pan_y + sample * self.cell_h
                x2 = self.pan_x + (trace + step_trace) * self.cell_w + 1
                y2 = self.pan_y + (sample + step_sample) * self.cell_h + 1
                if x2 < 0 or y2 < 0 or x1 > width or y1 > height:
                    continue
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color, tags=("map",))

        for item in self.results:
            x = self.pan_x + item.trace_index * self.cell_w
            y = self.pan_y + item.sample_index * self.cell_h
            if x < -20 or y < -20 or x > width + 20 or y > height + 20:
                continue
            bx1 = self.pan_x + item.trace_from * self.cell_w
            by1 = self.pan_y + item.sample_from * self.cell_h
            bx2 = self.pan_x + (item.trace_to + 1) * self.cell_w
            by2 = self.pan_y + (item.sample_to + 1) * self.cell_h
            self.canvas.create_rectangle(
                bx1,
                by1,
                bx2,
                by2,
                outline="#ffd166",
                width=2,
                dash=(5, 3),
                tags=("map",),
            )
            size = 7
            self.canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                outline="#ffd166",
                width=2,
                tags=("map",),
            )
            self.canvas.create_line(x - 10, y, x + 10, y, fill="#17212b", width=1, tags=("map",))
            self.canvas.create_line(x, y - 10, x, y + 10, fill="#17212b", width=1, tags=("map",))

        self.canvas.create_text(
            12,
            12,
            anchor="nw",
            text=f"ось X: трассы; ось Y: время/глубина; масштаб: {self.zoom:.1f}x",
            fill="#111111",
            font=("Segoe UI", 10, "bold"),
            tags=("overlay",),
        )
        self.canvas.create_text(
            12,
            34,
            anchor="nw",
            text="колесо мыши - масштаб, зажмите левую кнопку - перемещение",
            fill="#273746",
            font=("Segoe UI", 9),
            tags=("overlay",),
        )

    @staticmethod
    def display_scale(data):
        sampled = []
        row_step = max(1, len(data) // 500)
        col_step = max(1, len(data[0]) // 500)
        for row in data[::row_step]:
            sampled.extend(abs(value) for value in row[::col_step])
        if not sampled:
            return 1
        sampled.sort()
        index = min(len(sampled) - 1, int(len(sampled) * 0.985))
        scale = sampled[index]
        if not math.isfinite(scale) or scale <= 1e-9:
            scale = max(sampled) if sampled else 1
        return scale or 1

    @staticmethod
    def value_to_color(value):
        value = max(-1, min(1, value))
        base = (246, 248, 251)
        if value >= 0:
            target = (31, 182, 166)
            ratio = value
        else:
            target = (232, 69, 69)
            ratio = abs(value)
        red = int(base[0] + (target[0] - base[0]) * ratio)
        green = int(base[1] + (target[1] - base[1]) * ratio)
        blue = int(base[2] + (target[2] - base[2]) * ratio)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def draw_trace(self, trace_index):
        if not self.processed_data:
            return
        self.trace_canvas.delete("all")
        width = max(1, self.trace_canvas.winfo_width())
        height = max(1, self.trace_canvas.winfo_height())
        row = self.processed_data[trace_index]
        max_abs = max(abs(value) for value in row) or 1
        mid = height / 2

        for part in range(1, 4):
            y = height * part / 4
            self.trace_canvas.create_line(0, y, width, y, fill="#e1e7ee")
        self.trace_canvas.create_line(0, mid, width, mid, fill="#95a3b1")
        points = []
        for sample, value in enumerate(row):
            x = sample / max(1, len(row) - 1) * width
            y = mid - (value / max_abs) * (height * 0.42)
            points.extend([x, y])
        if len(points) >= 4:
            self.trace_canvas.create_line(*points, fill="#159184", width=2)

        for item in self.results:
            if item.trace_index == trace_index:
                x = item.sample_index / max(1, len(row) - 1) * width
                self.trace_canvas.create_line(x, 0, x, height, fill="#e84545", dash=(4, 3), width=2)

        self.trace_canvas.create_text(
            8,
            8,
            anchor="nw",
            text=f"Трасса {trace_index}",
            fill="#111111",
            font=("Segoe UI", 10, "bold"),
        )

    def start_pan(self, event):
        self.drag_start = (event.x, event.y, self.pan_x, self.pan_y)
        self.drag_moved = False

    def pan_heatmap(self, event):
        if self.drag_start is None:
            return
        start_x, start_y, original_x, original_y = self.drag_start
        dx = event.x - start_x
        dy = event.y - start_y
        if abs(dx) > 3 or abs(dy) > 3:
            self.drag_moved = True
        previous_x = self.pan_x
        previous_y = self.pan_y
        self.pan_x = original_x + dx
        self.pan_y = original_y + dy
        self.clamp_view()
        move_x = self.pan_x - previous_x
        move_y = self.pan_y - previous_y
        if move_x or move_y:
            self.canvas.move("map", move_x, move_y)

    def end_pan(self, event):
        if not self.drag_moved:
            self.on_canvas_click(event)
        else:
            self.draw_heatmap()
        self.drag_start = None
        self.drag_moved = False

    def zoom_heatmap(self, event):
        if not self.processed_data:
            return
        old_zoom = self.zoom
        direction = 1
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            direction = -1

        factor = 1.18 if direction > 0 else 1 / 1.18
        self.zoom = max(1.0, min(8.0, self.zoom * factor))
        if self.zoom == old_zoom:
            return

        scale = self.zoom / old_zoom
        self.pan_x = event.x - (event.x - self.pan_x) * scale
        self.pan_y = event.y - (event.y - self.pan_y) * scale
        self.clamp_view()
        self.draw_heatmap()

    def on_canvas_click(self, event):
        if not self.processed_data:
            return
        trace = int((event.x - self.pan_x) / max(self.cell_w, 1e-9))
        trace = max(0, min(len(self.processed_data) - 1, trace))
        self.draw_trace(trace)

    def on_tree_select(self, _event):
        selected = self.tree.selection()
        if not selected:
            return
        item = self.results[int(selected[0])]
        self.draw_trace(item.trace_index)

    def save_report(self):
        if not self.raw_data:
            messagebox.showwarning(
                "Нет данных для отчета",
                "Сначала загрузите файл или создайте демонстрационные сигналы."
            )
            return
        if not self.processed_data:
            messagebox.showwarning(
                "Анализ не выполнен",
                "Сначала выполните анализ, затем сохраните отчет."
            )
            return
        if not self.results:
            should_save = messagebox.askyesno(
                "Аномалии не найдены",
                "При текущих настройках зоны интереса не обнаружены.\n\n"
                "Сохранить отчет с нулевым результатом?"
            )
            if not should_save:
                return

        path = filedialog.asksaveasfilename(
            title="Сохранить отчет",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return

        lines = [
            "ОТЧЕТ ПО ОПРЕДЕЛЕНИЮ АНОМАЛИЙ ГЕОРАДИОЛОКАЦИОННЫХ СИГНАЛОВ",
            "",
            f"Источник данных: {self.source_info}",
            f"Количество трасс: {len(self.raw_data)}",
            f"Количество отсчетов в трассе: {len(self.raw_data[0])}",
            f"Выбранная антенна: {self.antenna_var.get()}",
            f"Максимальная расчетная глубина: {self.max_depth_m():.2f} м",
            f"Количество GPS-точек: {len(self.gps_points)}",
            f"Порог аномальности: {self.threshold_var.get():.2f}",
            f"Удаление фона: {'да' if self.bg_var.get() else 'нет'}",
            f"Компенсация затухания: {'да' if self.gain_var.get() else 'нет'}",
            f"Сглаживание: {'да' if self.smooth_var.get() else 'нет'}",
            "",
            "Найденные перспективные зоны:",
        ]
        for number, item in enumerate(self.results, start=1):
            lines.append(
                f"{number}. пик: трасса={item.trace_index}, отсчет={item.sample_index}; "
                f"границы: трассы {item.trace_from}-{item.trace_to}, отсчеты {item.sample_from}-{item.sample_to}; "
                f"глубина={item.depth_m:.2f} м; координаты={item.coordinate_text}; "
                f"score={item.score:.2f}, уверенность={item.confidence * 100:.0f}%, "
                f"амплитуда={item.amplitude:.4f}, тип={item.label}"
            )
            lines.append(f"   Обоснование: {item.reason}")

        if not self.results:
            lines.append("Перспективные зоны с заданным порогом не обнаружены.")

        lines.extend(
            [
                "",
                "Важно: результат является предварительной интерпретацией GPR-данных.",
                "Для принятия решения о раскопках требуется сопоставление с несколькими профилями,",
                "геодезической привязкой, глубинной калибровкой и экспертной археологической оценкой.",
            ]
        )

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.status_var.set(f"Отчет сохранен: {Path(path).name}")
        self.show_success("Отчет сохранен", f"Отчет успешно сохранен:\n{path}")


if __name__ == "__main__":
    app = GPRApp()
    app.mainloop()
