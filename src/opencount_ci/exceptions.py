# src/opencount_ci/exceptions.py
"""Custom exceptions for OpenCount CI."""


class OpenCountError(Exception):
    """Base exception for OpenCount CI."""


class ImageLoadError(OpenCountError):
    """Raised when an image cannot be loaded or validated."""


class DetectionError(OpenCountError):
    """Raised when detection fails."""


class ConfigurationError(OpenCountError):
    """Raised when configuration is invalid."""


class ValidationError(OpenCountError):
    """Raised when validation fails."""