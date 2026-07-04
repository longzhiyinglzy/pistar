"""Compatibility hooks for Hugging Face datasets imports.

LeRobot v2.1 parquet metadata can store list features as ``{"_type": "List"}``.
Some versions of ``datasets`` only register ``Sequence``/``LargeList`` and fail
while reading the Arrow schema before any project code sees a sample. Data
loaders import this module for side effects, so keep the fix here instead of
duplicating it in individual scripts.
"""

from __future__ import annotations

import json

from datasets.features.features import Features, _FEATURE_TYPES


if "List" not in _FEATURE_TYPES and "Sequence" in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = _FEATURE_TYPES["Sequence"]


_ORIGINAL_FROM_ARROW_SCHEMA = Features.from_arrow_schema


def _replace_legacy_list_features(feature_dict):
    for value in feature_dict.values():
        if isinstance(value, dict):
            if value.get("_type") == "List":
                value["_type"] = "Sequence"
            _replace_legacy_list_features(value)


def _patched_from_arrow_schema(arrow_schema):
    metadata = getattr(arrow_schema, "metadata", None)
    if metadata and b"info" in metadata:
        try:
            patched_metadata = dict(metadata)
            info = json.loads(patched_metadata[b"info"])
            if "features" in info:
                _replace_legacy_list_features(info["features"])
                patched_metadata[b"info"] = json.dumps(info).encode()
                arrow_schema = arrow_schema.with_metadata(patched_metadata)
        except Exception:
            # Fall back to the original parser so unrelated schema errors stay visible.
            pass
    return _ORIGINAL_FROM_ARROW_SCHEMA(arrow_schema)


if Features.from_arrow_schema is not _patched_from_arrow_schema:
    Features.from_arrow_schema = _patched_from_arrow_schema
