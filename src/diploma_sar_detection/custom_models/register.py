from __future__ import annotations


def register_context_modules() -> None:
    """Expose repository-local context modules to Ultralytics YAML parsing."""

    from ultralytics.nn import tasks

    from .adaptive_detail_gated_swin_fusion import AdaptiveDetailGatedSwinFusion
    from .gated_swin_fusion import GatedSwinFusion
    from .gated_wavevit_fusion import GatedWaveVitFusion
    from .residual_adaptive_swin_fusion import ResidualAdaptiveSwinFusion
    from .residual_adaptive_wavevit_fusion import ResidualAdaptiveWaveVitFusion
    from .residual_swin_c2psa import ResidualSwinC2PSA
    from .swin_context_block import SwinContextBlock
    from .wavevit_context_block import WaveVitContextBlock

    tasks.AdaptiveDetailGatedSwinFusion = AdaptiveDetailGatedSwinFusion
    tasks.GatedSwinFusion = GatedSwinFusion
    tasks.GatedWaveVitFusion = GatedWaveVitFusion
    tasks.ResidualAdaptiveSwinFusion = ResidualAdaptiveSwinFusion
    tasks.ResidualAdaptiveWaveVitFusion = ResidualAdaptiveWaveVitFusion
    tasks.ResidualSwinC2PSA = ResidualSwinC2PSA
    tasks.SwinContextBlock = SwinContextBlock
    tasks.WaveVitContextBlock = WaveVitContextBlock


def register_backbone(variant: str = "cnn_swin_t") -> None:
    """
    Monkey-patch Ultralytics `TorchVision` symbol with a local custom backbone.

    Supported variants:
    - `swin_t`: pure timm Swin-T wrapper
    - `cnn_swin_t`: lightweight CNN stem followed by timm Swin-T
    - `wavevit_s`/`wavevit_b`/`wavevit_l`: pure local WaveViT wrapper
    - `original_wavevit_s`/`original_wavevit_b`/`original_wavevit_l`: original-architecture WaveViT wrapper
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

    if normalized_variant in {"wavevit_s", "wavevit_b", "wavevit_l"}:
        from .wavevit_backbone import WaveVitBackbone

        tasks.TorchVision = WaveVitBackbone
        return

    if normalized_variant in {"original_wavevit_s", "original_wavevit_b", "original_wavevit_l"}:
        from .original_wavevit_backbone import OriginalWaveVitBackbone

        tasks.TorchVision = OriginalWaveVitBackbone
        return

    raise ValueError(
        f"Unsupported backbone variant: {variant}. "
        "Use 'swin_t', 'cnn_swin_t', 'wavevit_s', 'wavevit_b', 'wavevit_l', "
        "'original_wavevit_s', 'original_wavevit_b', or 'original_wavevit_l'."
    )


def register_swin_t_backbone() -> None:
    """Backward-compatible alias for registering the pure Swin-T wrapper."""

    register_backbone("swin_t")


def register_cnn_swin_t_backbone() -> None:
    """Register the hybrid CNN+Swin-T backbone wrapper."""

    register_backbone("cnn_swin_t")


def register_wavevit_backbone(variant: str = "wavevit_s") -> None:
    """Register a pure WaveViT backbone wrapper."""

    register_backbone(variant)
