from __future__ import annotations


def register_backbone(variant: str = "cnn_swin_t") -> None:
    """
    Monkey-patch Ultralytics `TorchVision` symbol with a local custom backbone.

    Supported variants:
    - `swin_t`: pure timm Swin-T wrapper
    - `cnn_swin_t`: lightweight CNN stem followed by timm Swin-T
    """

    normalized_variant = variant.strip().lower()
    from ultralytics.nn import tasks

    if normalized_variant == "swin_t":
        from .swin_t_backbone import SwinTBackbone

        tasks.TorchVision = SwinTBackbone
        return

    if normalized_variant == "cnn_swin_t":
        from .hybrid_cnn_swin_t_backbone import HybridCnnSwinTBackbone

        tasks.TorchVision = HybridCnnSwinTBackbone
        return

    raise ValueError(f"Unsupported backbone variant: {variant}. Use 'swin_t' or 'cnn_swin_t'.")


def register_swin_t_backbone() -> None:
    """Backward-compatible alias for registering the pure Swin-T wrapper."""

    register_backbone("swin_t")


def register_cnn_swin_t_backbone() -> None:
    """Register the hybrid CNN+Swin-T backbone wrapper."""

    register_backbone("cnn_swin_t")
