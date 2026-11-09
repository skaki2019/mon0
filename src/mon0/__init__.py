# Import the compiled nanobind module (private)
from ._impl import CSRDataset, TransitionMatrix0, TransitionMatrix, MultiOrderNetwork

# # Explicitly declare the public API

__all__ = ["CSRDataset", "TransitionMatrix0", "TransitionMatrix", "MultiOrderNetwork"]
# # Optional: expose a version
__version__ = "0.1.0"