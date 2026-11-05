"""viva_hybrid composite generators (imported for @composite_generator side effects)."""
from . import regime  # noqa: F401
from .regime import rust_hybrid_benchmark

__all__ = ["rust_hybrid_benchmark"]
