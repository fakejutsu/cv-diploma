from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn.tasks import DetectionModel


_DISTILL_RUNTIME: dict[int, dict[str, Any]] = {}


def _build_distill_loss(loss_name: str) -> nn.Module:
    normalized = loss_name.strip().lower()
    if normalized == "l1":
        return nn.L1Loss()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "smoothl1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported distill loss: {loss_name}. Use one of: l1, mse, smoothl1.")


class DistillSwinP5DetectionModel(DetectionModel):
    """DetectionModel that adds teacher-guided feature distillation on the fused P5 feature."""

    def __init__(
        self,
        cfg: str = "yolo26n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        distill_student_layer: int = 13,
        distill_student_channels: int = 256,
        distill_teacher_layer: int = 10,
        distill_teacher_channels: int = 512,
        distill_weight: float = 0.1,
        distill_loss: str = "smoothl1",
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.distill_student_layer = int(distill_student_layer)
        self.distill_teacher_layer = int(distill_teacher_layer)
        self.distill_student_channels = int(distill_student_channels)
        self.distill_teacher_channels = int(distill_teacher_channels)
        self.distill_weight = float(distill_weight)
        self.distill_loss_name = distill_loss
        self.distill_adapter = nn.Conv2d(self.distill_student_channels, self.distill_teacher_channels, kernel_size=1)
        self.distill_criterion = _build_distill_loss(self.distill_loss_name)
        object.__setattr__(self, "_distill_runtime_key", id(self))
        self._clear_runtime()

    def _clear_runtime(self) -> None:
        _DISTILL_RUNTIME[self._distill_runtime_key] = {
            "teacher_ref": None,
            "teacher_path": None,
        }

    def set_distillation_teacher(self, teacher_model: DetectionModel, teacher_path: str | None = None) -> None:
        runtime = _DISTILL_RUNTIME.setdefault(self._distill_runtime_key, {})
        runtime["teacher_ref"] = weakref.ref(teacher_model)
        runtime["teacher_path"] = teacher_path

    def clear_distillation_teacher(self) -> None:
        self._clear_runtime()

    def _teacher_model(self) -> DetectionModel | None:
        runtime = _DISTILL_RUNTIME.get(self._distill_runtime_key, {})
        teacher_ref = runtime.get("teacher_ref")
        return teacher_ref() if teacher_ref else None

    @staticmethod
    def _capture_layer_output(layer: nn.Module, forward_fn) -> tuple[Any, Tensor]:
        captured: dict[str, Tensor] = {}

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
            captured["value"] = output

        handle = layer.register_forward_hook(hook)
        try:
            result = forward_fn()
        finally:
            handle.remove()

        feature = captured.get("value")
        if feature is None:
            raise RuntimeError("Failed to capture the requested layer output for distillation.")
        return result, feature

    def _extract_teacher_feature(self, x: Tensor) -> Tensor:
        teacher = self._teacher_model()
        if teacher is None:
            raise RuntimeError("Distillation teacher is not configured.")

        teacher.eval()
        with torch.no_grad():
            _, feature = self._capture_layer_output(
                teacher.model[self.distill_teacher_layer],
                lambda: teacher.predict(x),
            )
        return feature.detach()

    def _extract_student_feature(self, x: Tensor, preds: Any | None = None) -> tuple[Any, Tensor]:
        if preds is None:
            preds, feature = self._capture_layer_output(
                self.model[self.distill_student_layer],
                lambda: self.forward(x),
            )
            return preds, feature

        _, feature = self._capture_layer_output(
            self.model[self.distill_student_layer],
            lambda: self.forward(x),
        )
        return preds, feature

    def compute_distillation_alignment(self, x: Tensor) -> dict[str, Tensor]:
        preds, student_feature = self._extract_student_feature(x)
        teacher_feature = self._extract_teacher_feature(x)
        adapted_student = self.distill_adapter(student_feature)
        if adapted_student.shape[-2:] != teacher_feature.shape[-2:]:
            teacher_feature = F.interpolate(
                teacher_feature,
                size=adapted_student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        distill_loss = self.distill_criterion(adapted_student.float(), teacher_feature.float())
        return {
            "preds": preds,
            "student_feature": student_feature,
            "teacher_feature": teacher_feature,
            "adapted_student_feature": adapted_student,
            "distill_loss": distill_loss,
        }

    def loss(self, batch: dict[str, Tensor], preds: Any | None = None) -> tuple[Tensor, Tensor]:
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds, student_feature = self._extract_student_feature(batch["img"], preds)
        det_loss, det_items = self.criterion(preds, batch)
        zero = det_items.new_tensor(0.0)
        teacher = self._teacher_model()
        if teacher is None or self.distill_weight <= 0.0:
            return det_loss, torch.cat((det_items, zero.unsqueeze(0)))

        teacher_feature = self._extract_teacher_feature(batch["img"])
        adapted_student = self.distill_adapter(student_feature)
        if adapted_student.shape[-2:] != teacher_feature.shape[-2:]:
            teacher_feature = F.interpolate(
                teacher_feature,
                size=adapted_student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        distill_loss = self.distill_criterion(adapted_student.float(), teacher_feature.float())
        batch_size = batch["img"].shape[0]
        total_loss = det_loss + self.distill_weight * distill_loss * batch_size
        return total_loss, torch.cat((det_items, distill_loss.detach().unsqueeze(0)))
