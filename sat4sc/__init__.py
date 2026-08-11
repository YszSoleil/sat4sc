"""sat4sc: spatial analysis tools for single-cell spatial transcriptomics."""

from . import pysphere, pysphere_plotting

# Backward-compatible attribute alias. The module file itself is now
# ``pysphere_plotting.py``; new code should import ``pysphere_plotting``.

__version__ = "0.2.1"

__all__ = ["pysphere", "pysphere_plotting"]
