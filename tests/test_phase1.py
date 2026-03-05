"""
Sanity-check tests for Phase 1 — Age Classification.

Run:  python -m pytest tests/test_phase1.py -v
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_class import build_model


def test_build_model_returns_module():
    """build_model should return an nn.Module."""
    model = build_model(num_classes=2)
    assert isinstance(model, nn.Module)


def test_output_shape():
    """Forward pass on a batch of 4 images should produce (4, 2) logits."""
    model = build_model(num_classes=2)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"


def test_output_shape_single():
    """Forward pass on a single image should produce (1, 2) logits."""
    model = build_model(num_classes=2)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"


def test_save_load_full_model():
    """Model saved with torch.save(model, ...) should reload as nn.Module."""
    model = build_model(num_classes=2)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(model, tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert isinstance(loaded, nn.Module), "Loaded object is not nn.Module"
        loaded.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, 2)
    finally:
        os.unlink(tmp_path)


def test_model_has_resnet18_backbone():
    """Model should contain ResNet-18 layers (layer1 through layer4)."""
    model = build_model(num_classes=2)
    assert hasattr(model, "layer1"), "Missing layer1"
    assert hasattr(model, "layer2"), "Missing layer2"
    assert hasattr(model, "layer3"), "Missing layer3"
    assert hasattr(model, "layer4"), "Missing layer4"


def test_model_parameter_count():
    """ResNet-18 with 2-class head should have ~11M parameters."""
    model = build_model(num_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < n_params < 15_000_000, (
        f"Unexpected parameter count: {n_params:,}"
    )


def test_gradient_flow():
    """A single forward + backward pass should produce non-zero gradients."""
    model = build_model(num_classes=2)
    model.train()
    x = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "No gradients flowed through the model"
