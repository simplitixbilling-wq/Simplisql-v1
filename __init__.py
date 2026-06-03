"""
SimplSQL Modules Package
------------------------
Modular components for the SimplSQL application.

Modules:
- ai: AI assistant and provider integrations
- ui: User interface components and dialogs
- core: Database operations and data loading
- transforms: Data transformation dialogs
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "SimplSQL Team"

# Package-level imports for convenience
from . import ai
from . import ui
from . import core
from . import transforms
from . import utils

__all__ = ['ai', 'ui', 'core', 'transforms', 'utils']
