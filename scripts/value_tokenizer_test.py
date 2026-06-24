import numpy as np
import orbax.checkpoint as ocp

from scripts import label_advantage_from_vlm
from scripts import train_value


class _TreeMetadataLike:
    """Matches Orbax TreeMetadata's mapping API without inheriting dict."""

    def __init__(self, tree):
        self._tree = tree

    def keys(self):
        return self._tree.keys()

    def __getitem__(self, key):
        return self._tree[key]


class _FakeGemmaTokenizer:
    def encode(self, text, *, add_bos, add_eos):
        assert text == "assemble the block\nValue:"
        assert add_bos is True
        assert add_eos is False
        return [1, 2, 3]


def _check_tokenizer(tokenizer):
    tokenizer._tokenizer = _FakeGemmaTokenizer()
    tokens, mask = tokenizer.tokenize(
        "assemble the block",
        state=np.zeros(7),
        adv_ind="positive",
        adv_ind_dropout=False,
    )
    np.testing.assert_array_equal(tokens, [1, 2, 3, 0, 0, 0])
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])


def test_train_value_tokenizer_accepts_transform_arguments():
    _check_tokenizer(train_value.GemmaValueTokenizer(max_len=6))


def test_label_value_tokenizer_accepts_transform_arguments():
    _check_tokenizer(label_advantage_from_vlm.GemmaValueTokenizer(max_len=6))


def test_jax_step_converts_to_python_int_for_tqdm():
    step = train_value.jnp.asarray(7)

    assert train_value._as_host_int(step) == 7
    assert isinstance(train_value._as_host_int(step), int)


def test_parquet_reader_skips_missing_columns_without_loading_images(tmp_path):
    path = tmp_path / "episode.parquet"
    label_advantage_from_vlm.pd.DataFrame(
        {
            "frame_index": [0, 1],
            "image": [b"frame-0", b"frame-1"],
        }
    ).to_parquet(path, index=False)

    result = label_advantage_from_vlm._read_parquet_columns(
        path,
        ["frame_index", "value_lable"],
    )

    assert list(result.columns) == ["frame_index"]
    assert result["frame_index"].tolist() == [0, 1]


def test_checkpoint_loader_selects_only_requested_parameter_tree(tmp_path):
    metadata = _TreeMetadataLike(
        {
            "params": {"weight": "regular"},
            "ema_params": {"weight": "ema"},
            "opt_state": {"weight": "optimizer"},
        }
    )
    checkpoint_path = tmp_path / "step_00000001"

    ema, ema_key = label_advantage_from_vlm._select_checkpoint_restore_item(
        metadata,
        use_ema=True,
        checkpoint_path=checkpoint_path,
    )
    regular, regular_key = label_advantage_from_vlm._select_checkpoint_restore_item(
        metadata,
        use_ema=False,
        checkpoint_path=checkpoint_path,
    )

    assert ema == {"ema_params": {"weight": "ema"}}
    assert ema_key == "ema_params"
    assert regular == {"params": {"weight": "regular"}}
    assert regular_key == "params"


def test_checkpoint_loader_enables_orbax_partial_restore(monkeypatch, tmp_path):
    captured = {}

    class _FakeCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def metadata(self, _path):
            return _TreeMetadataLike(
                {
                    "params": {"weight": "regular-metadata"},
                    "ema_params": {"weight": "ema-metadata"},
                }
            )

        def restore(self, _path, args):
            captured["args"] = args
            return {"ema_params": {"weight": np.asarray([2.0], dtype=np.float32)}}

    monkeypatch.setattr(ocp, "PyTreeCheckpointer", _FakeCheckpointer)

    params = label_advantage_from_vlm._load_checkpoint_params(
        tmp_path / "step_00000001",
        use_ema=True,
    )

    assert captured["args"].transforms == {}
    np.testing.assert_array_equal(np.asarray(params["weight"]), [2.0])
