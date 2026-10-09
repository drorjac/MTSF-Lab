import numpy as np
import torch
from dynforecast.training import seed_everything, train_network


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    seed_everything(33)
    x = np.arange(40, dtype=np.float32).reshape(20, 2) / 20
    y = x.sum(axis=1, keepdims=True)
    uninterrupted = torch.nn.Linear(2, 1)
    train_network(uninterrupted, x, y, x, y, epochs=4, batch_size=5, patience=20)
    seed_everything(33)
    partial = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "checkpoint.pt"
    train_network(partial, x, y, x, y, epochs=1, batch_size=5, patience=20, checkpoint=checkpoint)
    resumed = torch.nn.Linear(2, 1)
    diagnostics = train_network(
        resumed, x, y, x, y, epochs=4, batch_size=5, patience=20, checkpoint=checkpoint, resume=True
    )
    assert len(diagnostics["history"]) == 4
    for one, two in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
