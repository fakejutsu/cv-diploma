from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk

from sar_app.domain.entities import DetectionResult, ModelInfo
from sar_app.domain.interfaces import IModelRepository
from sar_app.scenario.object_detection_scenario import ObjectDetectionScenario


class MainWindow(tk.Tk):
    def __init__(
        self,
        model_repository: IModelRepository,
        scenario: ObjectDetectionScenario,
        demo_root: Path | None,
    ) -> None:
        super().__init__()
        self.title("SAR Object Detection")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self._model_repository = model_repository
        self._scenario = scenario
        self._demo_root = demo_root
        self._models = self._model_repository.list_models()
        self._selected_image: Path | None = None
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._display_image: ImageTk.PhotoImage | None = None
        self._is_running = False

        self._model_var = tk.StringVar()
        self._image_size_var = tk.StringVar()
        self._confidence_var = tk.DoubleVar(value=0.25)
        self._status_var = tk.StringVar(value="Готово")
        self._image_path_var = tk.StringVar(value="Изображение не выбрано")

        self._build_layout()
        self._populate_models()
        self.after(100, self._poll_result_queue)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(12, 10))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        ttk.Label(toolbar, text="Модель").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._model_combo = ttk.Combobox(toolbar, textvariable=self._model_var, state="readonly", width=34)
        self._model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self._model_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_model_changed())

        ttk.Label(toolbar, text="Image size").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self._image_size_combo = ttk.Combobox(toolbar, textvariable=self._image_size_var, state="readonly", width=10)
        self._image_size_combo.grid(row=0, column=3, sticky="w", padx=(0, 12))

        ttk.Label(toolbar, text="Confidence").grid(row=0, column=4, sticky="w", padx=(0, 8))
        confidence = ttk.Scale(toolbar, from_=0.05, to=0.95, variable=self._confidence_var, orient="horizontal")
        confidence.grid(row=0, column=5, sticky="ew", padx=(0, 8))
        toolbar.columnconfigure(5, weight=1)
        self._confidence_label = ttk.Label(toolbar, text="0.25", width=5)
        self._confidence_label.grid(row=0, column=6, sticky="w", padx=(0, 12))
        confidence.configure(command=lambda _value: self._update_confidence_label())

        controls = ttk.Frame(self, padding=(12, 0, 12, 8))
        controls.grid(row=1, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=2)
        controls.rowconfigure(1, weight=1)

        button_row = ttk.Frame(controls)
        button_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._choose_button = ttk.Button(
            button_row,
            text="Выбрать изображение",
            takefocus=False,
            command=lambda: self._defer_action(self._choose_image),
        )
        self._choose_button.pack(side="left", padx=(0, 8))
        self._demo_button = ttk.Button(
            button_row,
            text="Выбрать demo",
            takefocus=False,
            command=lambda: self._defer_action(self._choose_demo_image),
        )
        self._demo_button.pack(side="left", padx=(0, 8))
        self._run_button = ttk.Button(
            button_row,
            text="Запустить",
            takefocus=False,
            command=lambda: self._defer_action(self._run_detection),
        )
        self._run_button.pack(side="left", padx=(0, 12))
        ttk.Label(button_row, textvariable=self._image_path_var).pack(side="left", fill="x", expand=True)

        image_frame = ttk.Frame(controls, relief="solid", borderwidth=1)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self._image_label = ttk.Label(image_frame, text="Результирующее изображение появится здесь", anchor="center")
        self._image_label.grid(row=0, column=0, sticky="nsew")

        side_panel = ttk.Frame(controls)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=1)

        ttk.Label(side_panel, text="Описание результата").grid(row=0, column=0, sticky="w")
        self._summary_text = tk.Text(side_panel, wrap="word", height=18)
        self._summary_text.grid(row=1, column=0, sticky="nsew", pady=(4, 10))

        ttk.Label(side_panel, text="Статус").grid(row=2, column=0, sticky="w")
        ttk.Label(side_panel, textvariable=self._status_var, wraplength=360).grid(row=3, column=0, sticky="ew", pady=(4, 0))

    def _defer_action(self, action: Callable[[], None]) -> None:
        self._clear_button_visual_state()
        self.after_idle(action)

    def _clear_button_visual_state(self) -> None:
        for button in (self._choose_button, self._demo_button, self._run_button):
            button.state(("!pressed", "!active", "!focus"))
        self.focus_set()
        self.update_idletasks()

    def _populate_models(self) -> None:
        if not self._models:
            self._model_combo.configure(values=())
            self._image_size_combo.configure(values=())
            self._status_var.set("ONNX-модели не найдены. Добавьте .onnx и metadata.json в sar_app/models_repository.")
            return

        names = [model.name for model in self._models]
        self._model_combo.configure(values=names)
        self._model_var.set(names[0])
        self._on_model_changed()

    def _on_model_changed(self) -> None:
        model = self._selected_model()
        if model is None:
            return
        sizes = [str(size) for size in model.input_sizes]
        self._image_size_combo.configure(values=sizes)
        self._image_size_var.set(sizes[0])

    def _update_confidence_label(self) -> None:
        self._confidence_label.configure(text=f"{self._confidence_var.get():.2f}")

    def _choose_image(self) -> None:
        file_name = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=(
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ),
        )
        if file_name:
            self._selected_image = Path(file_name)
            self._image_path_var.set(str(self._selected_image))
        self._clear_button_visual_state()

    def _choose_demo_image(self) -> None:
        initial_dir = self._demo_root if self._demo_root and self._demo_root.exists() else Path.home()
        file_name = filedialog.askopenfilename(
            title="Выберите demo-изображение",
            initialdir=str(initial_dir),
            filetypes=(
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ),
        )
        if file_name:
            self._selected_image = Path(file_name)
            self._image_path_var.set(str(self._selected_image))
        self._clear_button_visual_state()

    def _run_detection(self) -> None:
        if self._is_running:
            return

        model = self._selected_model()
        if model is None:
            messagebox.showwarning("Нет модели", "Добавьте ONNX-модель и metadata.json.")
            return
        if self._selected_image is None:
            messagebox.showwarning("Нет изображения", "Выберите изображение для обнаружения.")
            return
        image_path = self._selected_image

        try:
            image_size = int(self._image_size_var.get())
        except ValueError:
            messagebox.showwarning("Некорректный размер", "Выберите допустимый image_size.")
            return

        confidence = float(self._confidence_var.get())
        self._is_running = True
        self._set_buttons_enabled(False)
        self._status_var.set("Выполняется обнаружение...")
        self.update_idletasks()

        self.after(50, lambda: self._start_detection_worker(model, image_path, image_size, confidence))

    def _start_detection_worker(
        self,
        model: ModelInfo,
        image_path: Path,
        image_size: int,
        confidence: float,
    ) -> None:
        thread = threading.Thread(
            target=self._run_detection_worker,
            args=(model, image_path, image_size, confidence),
            daemon=True,
        )
        thread.start()

    def _run_detection_worker(
        self,
        model: ModelInfo,
        image_path: Path,
        image_size: int,
        confidence: float,
    ) -> None:
        try:
            result = self._scenario.run(
                model=model,
                image_path=image_path,
                image_size=image_size,
                confidence=confidence,
            )
        except Exception as exc:
            self._result_queue.put(("error", exc))
        else:
            self._result_queue.put(("result", result))

    def _poll_result_queue(self) -> None:
        try:
            kind, payload = self._result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result_queue)
            return

        self._is_running = False
        self._set_buttons_enabled(True)
        self._clear_button_visual_state()

        if kind == "error":
            self._status_var.set(f"Ошибка: {payload}")
            messagebox.showerror("Ошибка обнаружения", str(payload))
        elif isinstance(payload, DetectionResult):
            self._show_result(payload)
            self._status_var.set("Готово")

        self.after(100, self._poll_result_queue)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (self._choose_button, self._demo_button, self._run_button):
            button.configure(state=state)

    def _show_result(self, result: DetectionResult) -> None:
        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("1.0", result.text_summary)

        image = result.rendered_image.image
        max_width = max(1, self._image_label.winfo_width() - 20)
        max_height = max(1, self._image_label.winfo_height() - 20)
        preview = image.copy()
        preview.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self._display_image = ImageTk.PhotoImage(preview)
        self._image_label.configure(image=self._display_image, text="")

    def _selected_model(self) -> ModelInfo | None:
        selected_name = self._model_var.get()
        for model in self._models:
            if model.name == selected_name:
                return model
        return None
