from __future__ import annotations

import weakref
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn.tasks import DetectionModel

from .distill_swin_p5_model import _build_distill_loss


_MULTI_DISTILL_RUNTIME: dict[int, dict[str, Any]] = {}


def _parse_int_sequence(values: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        return tuple(int(value.strip()) for value in values.split(",") if value.strip())
    return tuple(int(value) for value in values)


class DistillMultiFeatureDetectionModel(DetectionModel):
    """DetectionModel with teacher-guided distillation on multiple feature layers."""

    def __init__(
        self,
        cfg: str = "yolo26n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        distill_student_layers: Sequence[int] | str = (11, 12),
        distill_student_channels: Sequence[int] | str = (128, 256),
        distill_teacher_layers: Sequence[int] | str = (6, 10),
        distill_teacher_channels: Sequence[int] | str = (512, 512),
        distill_weight: float = 0.1,
        distill_loss: str = "smoothl1",
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.distill_student_layers = _parse_int_sequence(distill_student_layers)
        self.distill_student_channels = _parse_int_sequence(distill_student_channels)
        self.distill_teacher_layers = _parse_int_sequence(distill_teacher_layers)
        self.distill_teacher_channels = _parse_int_sequence(distill_teacher_channels)
        if not (
            len(self.distill_student_layers)
            == len(self.distill_student_channels)
            == len(self.distill_teacher_layers)
            == len(self.distill_teacher_channels)
        ):
            raise ValueError("Distillation layer/channel lists must have the same length.")
        if not self.distill_student_layers:
            raise ValueError("At least one distillation feature pair is required.")

        self.distill_weight = float(distill_weight)
        self.distill_loss_name = distill_loss
        self.distill_criterion = _build_distill_loss(self.distill_loss_name)
        self.distill_adapters = nn.ModuleDict(
            {
                str(student_layer): nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
                for student_layer, student_channels, teacher_channels in zip(
                    self.distill_student_layers,
                    self.distill_student_channels,
                    self.distill_teacher_channels,
                )
            }
        )

        object.__setattr__(self, "_distill_runtime_key", id(self))
        self._clear_runtime()

    def _clear_runtime(self) -> None:
        _MULTI_DISTILL_RUNTIME[self._distill_runtime_key] = {
            "teacher_ref": None,
            "teacher_path": None,
        }

    def set_distillation_teacher(self, teacher_model: DetectionModel, teacher_path: str | None = None) -> None:
        runtime = _MULTI_DISTILL_RUNTIME.setdefault(self._distill_runtime_key, {})
        runtime["teacher_ref"] = weakref.ref(teacher_model)
        runtime["teacher_path"] = teacher_path

    def clear_distillation_teacher(self) -> None:
        self._clear_runtime()

    def _teacher_model(self) -> DetectionModel | None:
        runtime = _MULTI_DISTILL_RUNTIME.get(self._distill_runtime_key, {})
        teacher_ref = runtime.get("teacher_ref")
        return teacher_ref() if teacher_ref else None

    @staticmethod
    def _capture_layer_outputs(
        layers: Sequence[nn.Module],
        forward_fn,
    ) -> tuple[Any, list[Tensor]]:
        captured: dict[int, Tensor] = {}
        handles = []

        for capture_index, layer in enumerate(layers):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor, index: int = capture_index) -> None:
                captured[index] = output

            handles.append(layer.register_forward_hook(hook))

        try:
            result = forward_fn()
        finally:
            for handle in handles:
                handle.remove()

        features = [captured.get(index) for index in range(len(layers))]
        if any(feature is None for feature in features):
            raise RuntimeError("Failed to capture all requested layer outputs for distillation.")
        return result, [feature for feature in features if feature is not None]

    def _extract_student_features(self, x: Tensor, preds: Any | None = None) -> tuple[Any, list[Tensor]]:
        layers = [self.model[layer_index] for layer_index in self.distill_student_layers]
        result, features = self._capture_layer_outputs(layers, lambda: self.forward(x))
        return preds if preds is not None else result, features

    def _extract_teacher_features(self, x: Tensor) -> list[Tensor]:
        teacher = self._teacher_model()
        if teacher is None:
            raise RuntimeError("Distillation teacher is not configured.")

        teacher.eval()
        teacher_parameter = next(teacher.parameters())
        teacher_input = x.to(device=teacher_parameter.device, dtype=teacher_parameter.dtype)
        layers = [teacher.model[layer_index] for layer_index in self.distill_teacher_layers]
        with torch.no_grad():
            with torch.amp.autocast(device_type=teacher_input.device.type, enabled=False):
                _, features = self._capture_layer_outputs(layers, lambda: teacher.predict(teacher_input))
        return [feature.detach() for feature in features]

    def _distill_loss(self, student_features: list[Tensor], teacher_features: list[Tensor]) -> Tensor:
        losses: list[Tensor] = []
        for student_layer, student_feature, teacher_feature in zip(
            self.distill_student_layers,
            student_features,
            teacher_features,
        ):
            adapted_student = self.distill_adapters[str(student_layer)](student_feature)
            if adapted_student.shape[-2:] != teacher_feature.shape[-2:]:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=adapted_student.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(self.distill_criterion(adapted_student.float(), teacher_feature.float()))

        return torch.stack(losses).mean()

    def compute_distillation_alignment(self, x: Tensor) -> dict[str, Any]:
        preds, student_features = self._extract_student_features(x)
        teacher_features = self._extract_teacher_features(x)
        distill_loss = self._distill_loss(student_features, teacher_features)
        return {
            "preds": preds,
            "student_features": student_features,
            "teacher_features": teacher_features,
            "distill_loss": distill_loss,
        }

    def loss(self, batch: dict[str, Tensor], preds: Any | None = None) -> tuple[Tensor, Tensor]:
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds, student_features = self._extract_student_features(batch["img"], preds)
        det_loss, det_items = self.criterion(preds, batch)
        zero = det_items.new_tensor(0.0)
        teacher = self._teacher_model()
        if teacher is None or self.distill_weight <= 0.0:
            return det_loss, torch.cat((det_items, zero.unsqueeze(0)))

        teacher_features = self._extract_teacher_features(batch["img"])
        distill_loss = self._distill_loss(student_features, teacher_features)
        batch_size = batch["img"].shape[0]
        total_loss = det_loss + self.distill_weight * distill_loss * batch_size
        return total_loss, torch.cat((det_items, distill_loss.detach().unsqueeze(0)))
