import numpy as np

from scripts import label_advantage_from_vlm
from scripts import train_value


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
