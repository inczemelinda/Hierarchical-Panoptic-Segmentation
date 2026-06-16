# Hiera backbone for Detectron2 / Mask2Former.
#
# Hiera (Ryali et al., ICML 2023 — "Hiera: A Hierarchical Vision Transformer
# without the Bells-and-Whistles") is a hierarchical Vision Transformer
# pretrained self-supervised with Masked Autoencoding (MAE).
#
# We wrap timm's Hiera implementation with features_only=True to get clean
# multi-scale feature maps (4 stages, strides 4/8/16/32). timm loads the MAE
# pretrained weights internally, so MODEL.WEIGHTS is left empty and the rest of
# the network (pixel decoder, transformer decoder) trains from scratch.
#
# Position-embedding note: hiera_large_224 uses a plain absolute pos_embed whose
# size is fixed at construction from img_size (and the unroll/reroll bookkeeping
# is tied to it too). timm does NOT interpolate it at runtime, and its pretrained
# checkpoint loader has the pos_embed resampling commented out. PyTorch's
# load_state_dict raises on a shape mismatch regardless of strict=, so we cannot
# let timm load pretrained into the 1024-sized model. The PhenoBench dataset
# mapper always feeds 1024x1024 images, so we:
#   1) build the feature extractor at img_size = HIERA.IMG_SIZE (default 1024)
#      WITHOUT pretrained (no automatic, crashing load),
#   2) fetch the pretrained MAE weights from a native (224) reference model,
#   3) resample the pretrained pos_embed to the 1024 grid,
#   4) load the resampled weights manually with strict=False.
# If resampling is unavailable we drop pos_embed and keep its random init; if the
# whole load fails we fall back to training the backbone from scratch.
import logging

import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

logger = logging.getLogger(__name__)


@BACKBONE_REGISTRY.register()
class D2Hiera(Backbone):
    """Hiera hierarchical ViT wrapped for Detectron2.

    Returns a dict of multi-scale features keyed res2/res3/res4/res5 to match
    the MSDeformAttn pixel decoder used by Mask2Former.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for the Hiera backbone. "
                "Install/upgrade with: pip install -U timm"
            ) from e

        model_name = cfg.MODEL.HIERA.NAME
        pretrained = cfg.MODEL.HIERA.PRETRAINED
        out_features = list(cfg.MODEL.HIERA.OUT_FEATURES)
        drop_path_rate = cfg.MODEL.HIERA.DROP_PATH_RATE
        img_size = cfg.MODEL.HIERA.IMG_SIZE          # must match the mapper output (1024)
        grad_checkpointing = cfg.MODEL.HIERA.GRAD_CHECKPOINTING

        # out_indices select the hierarchical stages (0..3 -> strides 4/8/16/32)
        out_indices = tuple(range(len(out_features)))

        # Build the feature extractor at the target size WITHOUT pretrained, so
        # timm does not try (and crash) to load the 224-sized pos_embed.
        create_kwargs = dict(
            pretrained=False,
            features_only=True,
            out_indices=out_indices,
            img_size=(img_size, img_size),
        )
        try:
            self.backbone = timm.create_model(
                model_name, drop_path_rate=drop_path_rate, **create_kwargs
            )
        except TypeError:
            # Some builds may not accept drop_path_rate through create_model.
            self.backbone = timm.create_model(model_name, **create_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create timm Hiera model '{model_name}' with "
                f"features_only=True. Ensure your timm version supports Hiera "
                f"feature extraction (pip install -U timm). Original error: {e}"
            ) from e

        # Manually load the MAE pretrained weights, resampling pos_embed to the
        # target grid first.
        if pretrained:
            self._load_pretrained(timm, model_name, img_size)

        if grad_checkpointing:
            underlying = getattr(self.backbone, "model", self.backbone)
            if hasattr(underlying, "set_grad_checkpointing"):
                underlying.set_grad_checkpointing(True)

        feat_info = self.backbone.feature_info
        self._out_features = out_features
        self._out_feature_channels = {
            name: feat_info[i]["num_chs"] for i, name in enumerate(out_features)
        }
        self._out_feature_strides = {
            name: feat_info[i]["reduction"] for i, name in enumerate(out_features)
        }
        self._size_divisibility = max(self._out_feature_strides.values())

    def _load_pretrained(self, timm, model_name, img_size):
        try:
            # Native (224) reference model loads its pretrained weights cleanly.
            ref = timm.create_model(model_name, pretrained=True)
            state = ref.state_dict()
            underlying = getattr(self.backbone, "model", self.backbone)
            dst_pos = getattr(underlying, "pos_embed", None)

            if (
                dst_pos is not None
                and "pos_embed" in state
                and state["pos_embed"].shape != dst_pos.shape
            ):
                try:
                    from timm.layers import resample_abs_pos_embed

                    ps = underlying.patch_stride
                    src_shape = tuple(state["pos_embed"].shape)
                    state["pos_embed"] = resample_abs_pos_embed(
                        state["pos_embed"],
                        new_size=(img_size // ps[0], img_size // ps[1]),
                        num_prefix_tokens=0,
                        verbose=False,
                    )
                    logger.info(
                        f"[D2Hiera] Resampled pretrained pos_embed "
                        f"{src_shape} -> {tuple(state['pos_embed'].shape)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[D2Hiera] pos_embed resample failed ({e}); dropping it "
                        f"(keeping random-init pos_embed)."
                    )
                    state.pop("pos_embed", None)

            missing, unexpected = underlying.load_state_dict(state, strict=False)
            logger.info(
                f"[D2Hiera] loaded MAE pretrained weights "
                f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
            )
            del ref
        except Exception as e:
            logger.warning(
                f"[D2Hiera] failed to load pretrained weights ({e}); "
                f"training the backbone from scratch."
            )

    def _to_nchw(self, feat, expected_chs):
        # timm Hiera forward_intermediates returns NCHW by default; convert
        # defensively just in case a build returns NHWC.
        if feat.dim() == 4 and feat.shape[1] != expected_chs and feat.shape[-1] == expected_chs:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        assert x.dim() == 4, f"Hiera takes an input of shape (N, C, H, W). Got {x.shape}."
        feats = self.backbone(x)  # list, one tensor per out_index
        outputs = {}
        for i, name in enumerate(self._out_features):
            outputs[name] = self._to_nchw(feats[i], self._out_feature_channels[name])
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility
