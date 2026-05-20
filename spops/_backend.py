"""Lazy access to the compiled native backends.

The CPU backend is always present; the CUDA backend is optional and may be
absent in CPU-only wheels. Callers should obtain backends through
:func:`get_cuda_backend` / :func:`get_cpu_backend` so that importing
:mod:`spops` does not fail on systems without CUDA.
"""

from __future__ import annotations

_cuda_backend = None
_cpu_backend = None


def get_cpu_backend():
    global _cpu_backend
    if _cpu_backend is None:
        import spops_backend_cpu  # noqa: WPS433

        _cpu_backend = spops_backend_cpu
    return _cpu_backend


def get_cuda_backend():
    global _cuda_backend
    if _cuda_backend is None:
        try:
            import spops_backend  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on build
            raise RuntimeError(
                "spops was built without the CUDA backend; GPU operations are "
                "unavailable. Rebuild spops on a machine with a CUDA toolkit "
                "(CUDA_HOME set) to enable them."
            ) from exc
        _cuda_backend = spops_backend
    return _cuda_backend
