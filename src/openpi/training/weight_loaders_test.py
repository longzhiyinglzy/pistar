import numpy as np

from openpi.training import weight_loaders


def test_convert_flat_gemma_checkpoint_remaps_mlp_weights():
    checkpoint = {
        "transformer/embedder": {"input_embedding": np.zeros((2, 3))},
        "transformer/layer_0/attn/q_einsum": {"w": np.zeros((2, 3))},
        "transformer/layer_0/mlp/gating_einsum": {"w": np.zeros((2, 3))},
        "transformer/layer_0/mlp/linear": {"w": np.zeros((3, 2))},
    }

    converted = weight_loaders._maybe_convert_gemma_ckpt_tree(checkpoint)

    assert converted["embedder"]["input_embedding"].shape == (2, 3)
    assert converted["layer_0"]["attn"]["q_einsum"]["w"].shape == (2, 3)
    assert converted["layer_0"]["mlp"]["gating_einsum"].shape == (2, 3)
    assert converted["layer_0"]["mlp"]["linear"].shape == (3, 2)
