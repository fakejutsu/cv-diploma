from __future__ import annotations


def register_swin_t_backbone() -> None:
    """
    Monkey-patch Ultralytics so YAMLs that reference `TorchVision` can resolve to the
    custom timm-based Swin-T wrapper during model construction.
    """

    from ultralytics.nn import tasks

    from .swin_t_backbone import SwinTBackbone

    tasks.TorchVision = SwinTBackbone
