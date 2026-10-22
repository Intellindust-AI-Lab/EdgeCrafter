"""Regression tests: empty-query classification losses in ECDet-S.

No pretrained weights, proprietary datasets or CUDA required.
"""
from types import SimpleNamespace

import pytest
import torch

from engine.core import YAMLConfig
from engine.edgecrafter.criterion import ECCriterion


def _criterion():
    return ECCriterion(
        matcher=SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={},
        losses=[],
        num_classes=3,
        alpha=0.2,
        gamma=1.5,
    )


def _case(query_count):
    logits = torch.tensor(
        [[[1.0, -0.4, -0.7], [0.3, 0.8, -0.2]]],
        requires_grad=True,
    )[:, :query_count, :]
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]
        )[:, :query_count],
    }
    n = 0 if query_count == 0 else 1
    targets = [{
        "labels": torch.zeros((n,), dtype=torch.long),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])[:n],
    }]
    indices = [(torch.arange(n), torch.arange(n))]
    return outputs, targets, indices


@pytest.mark.parametrize("loss_name", ["vfl", "mal"])
def test_zero_queries_produce_finite_raw_loss_and_backward(loss_name):
    outputs, targets, indices = _case(0)
    loss_fn = getattr(_criterion(), "loss_labels_" + loss_name)
    loss = next(iter(loss_fn(
        outputs, targets, indices, num_boxes=1.0,
        values=torch.empty(0),
    ).values()))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    (gradient,) = torch.autograd.grad(loss, outputs["pred_logits"])
    assert gradient.numel() == 0


@pytest.mark.parametrize("loss_name", ["vfl", "mal"])
def test_nonempty_queries_remain_finite(loss_name):
    outputs, targets, indices = _case(2)
    loss_fn = getattr(_criterion(), "loss_labels_" + loss_name)
    loss = next(iter(loss_fn(
        outputs, targets, indices, num_boxes=2.0,
        values=torch.tensor([0.75]),
    ).values()))
    assert torch.isfinite(loss)
    assert loss.item() > 0
    (gradient,) = torch.autograd.grad(loss, outputs["pred_logits"])
    assert torch.isfinite(gradient).all()


def test_normal_reduction_value_and_gradient_equivalence():
    torch.manual_seed(1)
    x = torch.randn(3, 7, 4, dtype=torch.float64, requires_grad=True)
    old = x.mean(1).sum() * x.shape[1] / 2.5
    new = x.sum() / 2.5
    torch.testing.assert_close(new, old, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        torch.autograd.grad(new, x, retain_graph=True)[0],
        torch.autograd.grad(old, x)[0],
        rtol=1e-13, atol=1e-13,
    )




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
    [(8, 1), (8, 0)],
    ids=["normal", "empty-targets"],
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
