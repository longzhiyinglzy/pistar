"""Small logging string helpers used by training scripts."""

from __future__ import annotations


def info(message: object) -> str:
    return f"[info] {message}"


def warn(message: object) -> str:
    return f"[warn] {message}"


def ok(message: object) -> str:
    return f"[ok] {message}"


def error(message: object) -> str:
    return f"[error] {message}"
