import random

import numpy as np
import torch


PAIRED_FORK_CHECKPOINT_VERSION = 1


def capture_rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": None,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint):
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    cuda_state = checkpoint.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_paired_fork_checkpoint(path, gaussians, iteration):
    checkpoint = {
        "version": PAIRED_FORK_CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "gaussian_state": gaussians.capture(),
        "confidence": gaussians.confidence,
        **capture_rng_state(),
    }
    torch.save(checkpoint, path)
    print(
        "[PairedForkSave] "
        f"iter={iteration} "
        f"gaussians={gaussians.get_xyz.shape[0]} "
        f"visibility_shape={_visibility_shape(gaussians)} "
        f"confidence_shape={_confidence_shape(gaussians.confidence)} "
        "optimizer_restored_capable=True",
        flush=True,
    )


def load_paired_fork_checkpoint(path, gaussians, training_args):
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("paired fork checkpoint must be a dict.")
    if checkpoint.get("version") != PAIRED_FORK_CHECKPOINT_VERSION:
        raise ValueError("unsupported paired fork checkpoint version.")

    gaussians.restore(
        checkpoint["gaussian_state"],
        training_args,
        restore_optimizer=True,
    )
    gaussians.confidence = _restore_confidence(checkpoint, gaussians)
    restore_rng_state(checkpoint)
    iteration = int(checkpoint["iteration"])
    print(
        "[PairedForkRestore] "
        f"iter={iteration} "
        f"gaussians={gaussians.get_xyz.shape[0]} "
        f"visibility_shape={_visibility_shape(gaussians)} "
        f"confidence_shape={_confidence_shape(gaussians.confidence)} "
        "optimizer_restored=True "
        "rng_restored=True",
        flush=True,
    )
    return iteration


def next_iteration_after_paired_fork(iteration):
    return int(iteration) + 1


def _visibility_shape(gaussians):
    history = getattr(gaussians, "visibility_history", None)
    if history is None:
        return None
    return tuple(history.shape)


def _confidence_shape(confidence):
    if confidence is None:
        return None
    return tuple(confidence.shape)


def _restore_confidence(checkpoint, gaussians):
    if "confidence" not in checkpoint:
        raise ValueError("paired fork checkpoint is missing confidence state.")

    confidence = checkpoint["confidence"]
    if not torch.is_tensor(confidence):
        raise ValueError("paired fork checkpoint confidence must be a tensor.")

    expected_count = gaussians.get_xyz.shape[0]
    if confidence.shape[0] != expected_count:
        raise ValueError(
            "paired fork checkpoint confidence first dimension "
            f"({confidence.shape[0]}) does not match Gaussian count ({expected_count})."
        )
    return confidence.to(device=gaussians.get_xyz.device, dtype=confidence.dtype)


def _torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
