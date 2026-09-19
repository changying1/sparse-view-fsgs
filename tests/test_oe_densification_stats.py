import pytest
import torch

from argparse import ArgumentParser
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.oe_densification import should_preserve_baseline_densification_stats


def _make_gaussian_model_for_stats(num_points):
    model = GaussianModel.__new__(GaussianModel)
    model.xyz_gradient_accum = torch.zeros((num_points, 1), dtype=torch.float32)
    model.denom = torch.zeros((num_points, 1), dtype=torch.float32)
    return model


def test_add_densification_stats_default_uses_viewspace_grad():
    model = _make_gaussian_model_for_stats(3)
    viewspace = torch.zeros((3, 3), dtype=torch.float32, requires_grad=True)
    total_grad = torch.tensor(
        [[3.0, 4.0, 100.0], [0.0, 2.0, 100.0], [5.0, 12.0, 100.0]],
        dtype=torch.float32,
    )
    viewspace.grad = total_grad
    update_filter = torch.tensor([True, False, True])

    model.add_densification_stats(viewspace, update_filter)

    assert torch.allclose(model.xyz_gradient_accum.squeeze(), torch.tensor([5.0, 0.0, 13.0]))
    assert torch.equal(model.denom.squeeze(), torch.tensor([1.0, 0.0, 1.0]))


def test_add_densification_stats_override_uses_baseline_grad_not_total_grad():
    model = _make_gaussian_model_for_stats(3)
    viewspace = torch.zeros((3, 3), dtype=torch.float32, requires_grad=True)
    viewspace.grad = torch.tensor(
        [[30.0, 40.0, 0.0], [0.0, 20.0, 0.0], [50.0, 120.0, 0.0]],
        dtype=torch.float32,
    )
    baseline_grad = torch.tensor(
        [[3.0, 4.0, 0.0], [7.0, 24.0, 0.0], [5.0, 12.0, 0.0]],
        dtype=torch.float32,
    )
    update_filter = torch.tensor([True, True, False])

    model.add_densification_stats(viewspace, update_filter, grad_override=baseline_grad)

    assert torch.allclose(model.xyz_gradient_accum.squeeze(), torch.tensor([5.0, 25.0, 0.0]))
    assert torch.equal(model.denom.squeeze(), torch.tensor([1.0, 1.0, 0.0]))


def test_add_densification_stats_override_preserves_denominator_counting():
    model = _make_gaussian_model_for_stats(2)
    viewspace = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    viewspace.grad = torch.ones_like(viewspace)
    baseline_grad = torch.full_like(viewspace, 2.0)
    update_filter = torch.tensor([True, False])

    model.add_densification_stats(viewspace, update_filter, grad_override=baseline_grad)
    model.add_densification_stats(viewspace, update_filter, grad_override=baseline_grad)

    assert torch.equal(model.denom.squeeze(), torch.tensor([2.0, 0.0]))


def test_add_densification_stats_invalid_override_shape_raises_value_error():
    model = _make_gaussian_model_for_stats(2)
    viewspace = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    viewspace.grad = torch.ones_like(viewspace)
    bad_baseline_grad = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="shape"):
        model.add_densification_stats(
            viewspace,
            torch.tensor([True, False]),
            grad_override=bad_baseline_grad,
        )


def test_baseline_autograd_grad_is_separate_from_total_backward_gradient():
    parameter = torch.tensor([2.0], requires_grad=True)
    viewspace = parameter * torch.tensor([1.0, 2.0, 3.0])
    viewspace.retain_grad()
    base_loss = viewspace[0] + 2.0 * viewspace[1]
    oe_loss = 3.0 * viewspace[2]
    total_loss = base_loss + oe_loss

    baseline_viewspace_grad = torch.autograd.grad(
        base_loss,
        viewspace,
        retain_graph=True,
        allow_unused=False,
    )[0].detach()
    viewspace.grad = None
    total_loss.backward()

    assert torch.allclose(baseline_viewspace_grad, torch.tensor([1.0, 2.0, 0.0]))
    assert torch.allclose(viewspace.grad, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(parameter.grad, torch.tensor([14.0]))
    assert torch.isfinite(baseline_viewspace_grad).all()
    assert torch.isfinite(viewspace.grad).all()


def test_preserve_densification_stats_path_requires_flag_oe_active_and_densify_window():
    assert should_preserve_baseline_densification_stats(True, True, 999, 1000) is True
    assert should_preserve_baseline_densification_stats(False, True, 999, 1000) is False
    assert should_preserve_baseline_densification_stats(True, False, 999, 1000) is False
    assert should_preserve_baseline_densification_stats(True, True, 1000, 1000) is False


def test_oe_preserve_baseline_densification_stats_cli_defaults_off_and_can_enable():
    parser = ArgumentParser()
    OptimizationParams(parser)

    default_args = parser.parse_args([])
    enabled_args = parser.parse_args(["--oe_preserve_baseline_densification_stats"])

    assert default_args.oe_preserve_baseline_densification_stats is False
    assert enabled_args.oe_preserve_baseline_densification_stats is True
