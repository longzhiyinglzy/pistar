"""Progress-bar helpers.

This module intentionally keeps the helper tiny. Some project scripts call
`sync_pbar_color` to share terminal styling with custom tqdm wrappers, but the
training logic must not depend on that styling being available.
"""

from __future__ import annotations


def sync_pbar_color(_pbar: object) -> None:
    return None
