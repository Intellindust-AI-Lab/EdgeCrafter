"""Regression tests: ECTransformer with num_denoising=0.

No pretrained weights, proprietary datasets or CUDA required.
"""
from types import MethodType

import pytest
import torch
import torch.nn as nn

from engine.core import YAMLConfig
from engine.edgecrafter.decoder import ECTransformer


class _StubDecoder(nn.Module):
    def __init__(self, with_masks):
        super().__init__()
        self.with_masks = with_masks
        self.pre_segs = None

    def forward(self, *args, **kwargs):
        layers, batch, queries, classes = 2, 1, 3, 3
        bboxes = torch.zeros(layers, batch, queries, 4)
        logits = torch.zeros(layers, batch, queries, classes)
        corners = torch.zeros(layers, batch, queries, 8)
        refs = torch.zeros(layers, batch, queries, 4)
        masks = torch.zeros(layers, batch, queries, 2, 2) if self.with_masks else None
        pre_bboxes = torch.zeros(batch, queries, 4)
        pre_logits = torch.zeros(batch, queries, classes)
        pre_segs = (
            torch.full((batch, queries, 2, 2), 3.0)
            if self.with_masks else None
        )
        self.pre_segs = pre_segs
        return bboxes, logits, corners, refs, masks, pre_bboxes, pre_logits, pre_segs


def _transformer_without_denoising(with_masks):
    # Exercise the actual ECTransformer.forward without constructing a ViT.
    model = ECTransformer.__new__(ECTransformer)
    nn.Module.__init__(model)
    model.num_denoising = 0
    model.aux_loss = True
    model.query_select_method = "default"
    model.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    model.reg_scale = nn.Parameter(torch.tensor([4.0]), requires_grad=False)
    model.decoder = _StubDecoder(with_masks)
    for name in ("dec_bbox_head", "dec_score_head", "query_pos_head",
                 "pre_bbox_head", "integral"):
        setattr(model, name, nn.Identity())

    def fake_encoder_input(self, feats):
        return torch.zeros(1, 4, 3), [[2, 2]]

    def fake_decoder_input(self, memory, shapes, denoising_logits, denoising_bbox_unact):
        assert denoising_logits is None and denoising_bbox_unact is None
        return (
            torch.zeros(1, 3, 3), torch.zeros(1, 3, 4),
            [torch.zeros(1, 3, 4)], [torch.zeros(1, 3, 3)],
        )

    model._get_encoder_input = MethodType(fake_encoder_input, model)
    model._get_decoder_input = MethodType(fake_decoder_input, model)
    model.train()
    return model


@pytest.mark.parametrize("with_masks", [False, True])
def test_train_without_denoising_preserves_segmentation_outputs(with_masks):
    model = _transformer_without_denoising(with_masks)
    outputs = model([torch.zeros(1, 3, 2, 2)], targets=None)
    assert "dn_outputs" not in outputs
    assert "dn_meta" not in outputs
    assert "pre_outputs" in outputs
    if with_masks:
        torch.testing.assert_close(
            outputs["pre_outputs"]["pred_masks"], model.decoder.pre_segs
        )
        assert outputs["pred_masks"] is not None
    else:
        assert outputs["pre_outputs"]["pred_masks"] is None
        assert outputs["pred_masks"] is None


@pytest.fixture(scope="module")
def training_setup():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    cfg = YAMLConfig("configs/ecdet/ecdet_s.yml")

    # Preserve the real ECDet-S backbone, encoder, transformer and criterion.
    # Reduce resolution/decoder workload so a hosted CPU runner can execute it.
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    cfg.yaml_cfg["ECTransformer"]["num_queries"] = 32
    cfg.yaml_cfg["ECTransformer"]["num_layers"] = 2
    cfg.yaml_cfg["ECTransformer"]["num_denoising"] = 8

    model = cfg.model.train()
    criterion = cfg.criterion.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return model, criterion, optimizer


@pytest.mark.parametrize(
    "num_denoising,num_objects",
    [(0, 1)],
    ids=["denoising-disabled"],
)
def test_real_model_forward_backward_and_optimizer(
    training_setup, num_denoising, num_objects
):
    model, criterion, optimizer = training_setup
    model.decoder.num_denoising = num_denoising
    targets = [{
        "labels": torch.zeros(num_objects, dtype=torch.long),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2]] * num_objects,
            dtype=torch.float32,
        ).reshape(-1, 4),
    }]

    optimizer.zero_grad(set_to_none=True)
    images = torch.randn(1, 3, 128, 128)
    outputs = model(images, targets=targets)
    assert "pred_boxes" in outputs
    assert "pred_logits" in outputs
    assert ("dn_outputs" in outputs) == (num_denoising > 0)

    losses = criterion(outputs, targets)
    assert losses, "Criterion produced no losses"
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name}: {value}"

    total_loss = sum(losses.values())
    assert torch.isfinite(total_loss), f"Nonfinite total loss: {total_loss}"
    total_loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in gradients), "Nonfinite gradient"
    optimizer.step()
