import numpy as np
import pytest

from openpi.policies import value_policy


def test_value_inputs_squeezes_scalar_like_value():
    transform = value_policy.ValueInputs()
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the block",
        "value": np.array([-0.5], dtype=np.float32),
    }

    transformed = transform(sample)

    assert np.asarray(transformed["value"]).shape == ()
    assert float(transformed["value"]) == pytest.approx(-0.5)


def test_value_inputs_rejects_non_scalar_value():
    transform = value_policy.ValueInputs()
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the block",
        "value": np.array([-0.5, -0.25], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="value must be scalar-like"):
        transform(sample)
