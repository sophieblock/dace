"""
Pytest configuration for TensorFlow tests.

These tests were written against TensorFlow v1-style APIs (tf.placeholder, tf.Session,
tf.ConfigProto, ...). On modern environments, TensorFlow v2 is typically installed,
where these symbols moved under tf.compat.v1 and eager execution is enabled by default.

This conftest provides a minimal compatibility shim so the tests run on TF2 without
requiring a separate environment:
- Disable eager execution.
- Expose common v1 symbols at the TensorFlow root module.

The shim is applied only when the root-level attributes are missing (i.e., TF2).
"""

from __future__ import annotations

import os


def _apply_tf_v1_compat_shim() -> None:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        # TensorFlow not installed; let tests skip/fail normally.
        return

    # If TF already exposes the v1 symbols at root (TF1), do nothing.
    if all(hasattr(tf, name) for name in ("placeholder", "Session", "ConfigProto")):
        return

    # Reduce TF C++ verbosity
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Enable TF1 graph mode behavior on TF2
    try:
        tf.compat.v1.disable_eager_execution()
        # Some installations also require this for full v1 semantics
        tf.compat.v1.disable_v2_behavior()
    except Exception:
        # If compat API unavailable, nothing more we can do
        pass

    # Provide minimal set of v1 symbols at the root module expected by tests
    try:
        if not hasattr(tf, "placeholder"):
            tf.placeholder = tf.compat.v1.placeholder  # type: ignore[attr-defined]
        if not hasattr(tf, "Session"):
            tf.Session = tf.compat.v1.Session  # type: ignore[attr-defined]
        if not hasattr(tf, "ConfigProto"):
            tf.ConfigProto = tf.compat.v1.ConfigProto  # type: ignore[attr-defined]
        # Control flow and random ops sometimes expected at root
        if not hasattr(tf, "random_uniform") and hasattr(tf.compat.v1, "random_uniform"):
            tf.random_uniform = tf.compat.v1.random_uniform  # type: ignore[attr-defined]
        if not hasattr(tf, "truncated_normal") and hasattr(tf.compat.v1, "truncated_normal"):
            tf.truncated_normal = tf.compat.v1.truncated_normal  # type: ignore[attr-defined]
        if not hasattr(tf, "random_normal") and hasattr(tf.compat.v1, "random_normal"):
            tf.random_normal = tf.compat.v1.random_normal  # type: ignore[attr-defined]
        # Expose nn submodule functions commonly used
        try:
            _nn = tf.nn
            if not hasattr(_nn, "top_k") and hasattr(tf.compat.v1.nn, "top_k"):
                _nn.top_k = tf.compat.v1.nn.top_k  # type: ignore[attr-defined]
            if not hasattr(_nn, "conv3d") and hasattr(tf.compat.v1.nn, "conv3d"):
                _nn.conv3d = tf.compat.v1.nn.conv3d  # type: ignore[attr-defined]
        except Exception:
            pass
        # A few extra common aliases that may be referenced indirectly
        if not hasattr(tf, "GraphDef") and hasattr(tf.compat.v1, "GraphDef"):
            tf.GraphDef = tf.compat.v1.GraphDef  # type: ignore[attr-defined]
        if not hasattr(tf, "InteractiveSession") and hasattr(tf.compat.v1, "InteractiveSession"):
            tf.InteractiveSession = tf.compat.v1.InteractiveSession  # type: ignore[attr-defined]
        if not hasattr(tf, "get_default_graph") and hasattr(tf.compat.v1, "get_default_graph"):
            tf.get_default_graph = tf.compat.v1.get_default_graph  # type: ignore[attr-defined]
        if not hasattr(tf, "reset_default_graph") and hasattr(tf.compat.v1, "reset_default_graph"):
            tf.reset_default_graph = tf.compat.v1.reset_default_graph  # type: ignore[attr-defined]
    except Exception:
        # Best-effort shim; allow tests to surface specific failures if any
        pass


# Apply shim as early as possible during test collection
_apply_tf_v1_compat_shim()
