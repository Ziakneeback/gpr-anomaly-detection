import csv
import math
import random
import statistics
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


SAMPLE_COUNT = 240
TRACE_COUNT = 96


@dataclass
class DetectionResult:
    trace_index: int
    sample_index: int
    score: float
    amplitude: float
    label: str


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
    def load_csv(path):
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
    def preprocess(data, remove_background=True, gain=True, smoothing=True):
        processed = [row[:] for row in data]
        if remove_background:
            processed = GPRProcessor.mean_center(processed)
        if gain:
            processed = GPRProcessor.gain_compensation(processed)
        if smoothing:
            processed = GPRProcessor.moving_average(processed)
        return processed

    @staticmethod
    def detect_anomalies(data, threshold=3.0, min_distance=7, max_results=25):
        values = [abs(value) for row in data for value in row]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations) or 1e-9

        candidates = []
        for trace_index, row in enumerate(data):
            for sample_index, value in enumerate(row):
                score = 0.6745 * (abs(value) - median) / mad
                if score >= threshold:
                    label = "сильное отражение" if value > 0 else "ослабление сигнала"
                    candidates.append(
                        DetectionResult(trace_index, sample_index, score, value, label)
                    )

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = []
        for item in candidates:
            too_close = any(
                abs(item.trace_index - prev.trace_index) <= min_distance
                and abs(item.sample_index - prev.sample_index) <= min_distance
                for prev in selected
            )
            if not too_close:
                selected.append(item)
            if len(selected) >= max_results:
                break
        return selected


class GPRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Определение аномалий георадиолокационных сигналов")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.raw_data = GPRProcessor.generate_demo_data()
        self.processed_data = []
        self.results = []
        self.cell_w = 1
        self.cell_h = 1
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = None
        self.drag_moved = False

        self.threshold_var = tk.DoubleVar(value=3.0)
        self.bg_var = tk.BooleanVar(value=True)
        self.gain_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Загружен демонстрационный набор данных")

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
            panel, text="Загрузить CSV", style="Accent.TButton", command=self.load_csv
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
            command=lambda _value: self.run_detection(),
        )
        threshold.pack(fill="x", pady=(4, 0))
        self.threshold_label = ttk.Label(panel, style="SideBody.TLabel")
        self.threshold_label.pack(anchor="w", pady=(0, 12))

        ttk.Checkbutton(
            panel,
            text="Удаление фоновой составляющей",
            variable=self.bg_var,
            command=self.run_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            panel,
            text="Компенсация затухания с глубиной",
            variable=self.gain_var,
            command=self.run_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            panel,
            text="Сглаживание шума",
            variable=self.smooth_var,
            command=self.run_detection,
            style="Panel.TCheckbutton",
        ).pack(anchor="w", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Button(
            panel, text="Выполнить анализ", style="Accent.TButton", command=self.run_detection
        ).pack(fill="x", pady=3)
        ttk.Button(
            panel, text="Сохранить отчет", style="Action.TButton", command=self.save_report
        ).pack(fill="x", pady=3)

        ttk.Separator(panel).pack(fill="x", pady=14)
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
        self.canvas.bind("<Configure>", lambda _event: self.draw_heatmap())

        right = tk.Frame(workspace, bg=bg)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Список аномалий", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        columns = ("trace", "sample", "score", "amp", "label")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=16)
        headings = {
            "trace": "Трасса",
            "sample": "Отсчет",
            "score": "Оценка",
            "amp": "Амплитуда",
            "label": "Тип",
        }
        widths = {"trace": 62, "sample": 64, "score": 70, "amp": 80, "label": 150}
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
        self.raw_data = GPRProcessor.generate_demo_data()
        self.reset_view(redraw=False)
        self.status_var.set("Создан новый демонстрационный набор георадиолокационных сигналов")
        self.run_detection()

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Выберите CSV с трассами ГРЛ",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.raw_data = GPRProcessor.load_csv(path)
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", str(exc))
            return
        self.reset_view(redraw=False)
        self.status_var.set(f"Загружен файл: {Path(path).name}")
        self.run_detection()

    def reset_view(self, redraw=True):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if redraw:
            self.draw_heatmap()

    def clamp_view(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
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

    def run_detection(self):
        if not self.raw_data:
            return
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
        self.refresh_table()
        self.draw_heatmap()
        self.draw_trace(self.results[0].trace_index if self.results else 0)
        self.status_var.set(
            f"Трасс: {len(self.raw_data)}, отсчетов: {len(self.raw_data[0])}, "
            f"найдено аномалий: {len(self.results)}"
        )

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
                    f"{item.score:.2f}",
                    f"{item.amplitude:.3f}",
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
        self.clamp_view()
        self.cell_w = (width / rows) * self.zoom
        self.cell_h = (height / cols) * self.zoom
        max_abs = max(abs(value) for row in self.processed_data for value in row) or 1

        step_trace = max(1, int(rows / (140 * self.zoom)))
        step_sample = max(1, int(cols / (260 * self.zoom)))
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
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color)

        for item in self.results:
            x = self.pan_x + item.trace_index * self.cell_w
            y = self.pan_y + item.sample_index * self.cell_h
            if x < -20 or y < -20 or x > width + 20 or y > height + 20:
                continue
            size = 7
            self.canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                outline="#ffd166",
                width=2,
            )
            self.canvas.create_line(x - 10, y, x + 10, y, fill="#17212b", width=1)
            self.canvas.create_line(x, y - 10, x, y + 10, fill="#17212b", width=1)

        self.canvas.create_text(
            12,
            12,
            anchor="nw",
            text=f"ось X: трассы; ось Y: время/глубина; масштаб: {self.zoom:.1f}x",
            fill="#111111",
            font=("Segoe UI", 10, "bold"),
        )
        self.canvas.create_text(
            12,
            34,
            anchor="nw",
            text="колесо мыши - масштаб, зажмите левую кнопку - перемещение",
            fill="#273746",
            font=("Segoe UI", 9),
        )

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
        self.pan_x = original_x + dx
        self.pan_y = original_y + dy
        self.clamp_view()
        self.draw_heatmap()

    def end_pan(self, event):
        if not self.drag_moved:
            self.on_canvas_click(event)
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
            f"Количество трасс: {len(self.raw_data)}",
            f"Количество отсчетов в трассе: {len(self.raw_data[0])}",
            f"Порог аномальности: {self.threshold_var.get():.2f}",
            f"Удаление фона: {'да' if self.bg_var.get() else 'нет'}",
            f"Компенсация затухания: {'да' if self.gain_var.get() else 'нет'}",
            f"Сглаживание: {'да' if self.smooth_var.get() else 'нет'}",
            "",
            "Найденные аномалии:",
        ]
        for number, item in enumerate(self.results, start=1):
            lines.append(
                f"{number}. трасса={item.trace_index}, отсчет={item.sample_index}, "
                f"оценка={item.score:.2f}, амплитуда={item.amplitude:.4f}, тип={item.label}"
            )

        if not self.results:
            lines.append("Аномалии с заданным порогом не обнаружены.")

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.status_var.set(f"Отчет сохранен: {Path(path).name}")


if __name__ == "__main__":
    app = GPRApp()
    app.mainloop()
