from functools import partial

import pytest
import torch

from src.models import FixedLengthModel, PerResidueModel

SEQUENCE_LEN = 30
EMBEDDING_SIZE = 60
MODEL_NAMES = ["cnn", "ffnn"]


def create_dummy_model(model_name):
    return {
        "cnn": partial(PerResidueModel, lr=0.01, weight_decay=0),
        "ffnn": partial(FixedLengthModel, num_hidden=10, lr=0.01, weight_decay=0),
    }[model_name]


@torch.no_grad()
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_forward_pass(model_name):
    # Confirm that forward pass returns a single logit
    dummy = torch.randn(1, SEQUENCE_LEN, EMBEDDING_SIZE)
    model = create_dummy_model(model_name)()
    logit = model(dummy)
    assert logit.shape == (1, 1)  # 1 sample, 1 logit

@torch.no_grad()
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_step(model_name):
    # Confirm the training/validation steps return a single value
    dummy_batch = (
        torch.randn(1, SEQUENCE_LEN, EMBEDDING_SIZE),
        torch.randint(high=2, size=(1, ))
    )
    model = create_dummy_model(model_name)()
    loss = model.step(dummy_batch, "training")
    assert loss.shape == ()  # 1 loss
