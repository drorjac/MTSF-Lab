import copy
import random
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def train_network(
    network,
    x,
    y,
    vx,
    vy,
    *,
    epochs=30,
    batch_size=32,
    lr=0.001,
    patience=8,
    device="cpu",
    deadline=None,
    checkpoint=None,
    resume=False,
    loss_fn=None,
    tensorboard_dir=None,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.set_device(torch.device(device))
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (x, y, vx, vy)]
    loader = DataLoader(TensorDataset(*tensors[:2]), batch_size=batch_size, shuffle=True)
    vx, vy = (a.to(device) for a in tensors[2:])
    history, best, stale, best_epoch, first_epoch = [], float("inf"), 0, -1, 0
    best_state = copy.deepcopy(network.state_dict())
    if resume and checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, stale = state["history"], state["best"], state["stale"]
        best_epoch, first_epoch = state["best_epoch"], state["epoch"] + 1
        best_state = state["best_state"]
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, epochs):
        if stale >= patience:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Global experiment budget exhausted; checkpoint retained")
        tick = time.perf_counter()
        network.train()
        total, count, grad = 0.0, 0, 0.0
        for bx, by in loader:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Global budget exhausted within epoch; resume from last complete epoch"
                )
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            prediction = network(bx)
            loss = (
                torch.nn.functional.mse_loss(prediction, by)
                if loss_fn is None
                else loss_fn(prediction, by, bx)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                network.parameters(), 5.0, error_if_nonfinite=True
            )
            grad = max(grad, float(norm))
            optimizer.step()
            total += float(loss.detach()) * len(bx)
            count += len(bx)
        network.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(network(vx), vy))
        if not np.isfinite(validation):
            raise FloatingPointError("Nonfinite validation loss")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "validation_loss": validation,
                "seconds": time.perf_counter() - tick,
                "gradient_norm": grad,
            }
        )
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            with SummaryWriter(str(tensorboard_dir)) as writer:
                for key in ("train_loss", "validation_loss", "gradient_norm", "seconds"):
                    writer.add_scalar(key, history[-1][key], epoch)
        if validation < best:
            best, stale, best_epoch = validation, 0, epoch
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if checkpoint is not None:
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "model": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "best": best,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "epoch": epoch,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.startswith("cuda")
                    else None,
                },
                temporary,
            )
            temporary.replace(checkpoint)
    network.load_state_dict(best_state)
    network.eval()
    return {
        "history": history,
        "best_epoch": best_epoch,
        "stop_epoch": len(history) - 1,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else 0,
        "training_seconds": sum(row["seconds"] for row in history),
    }
