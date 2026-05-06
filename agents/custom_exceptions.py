"""
Custom error codes used for custom built exceptions

IncompatibleFileFormatException: 700
EmptyFileExtensionException: 700.2
ImageEncodingFunctionException: 701
ImageBoundingBoxIdentificationException: 707
ImageBase64EncodingException: 714
"""

class ImageEncodingException(Exception):
    """Custom exception to handle image encoding exceptions during document parsing

    Args:
        Exception: Inheriting from the base Exception class
    """

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code
    
    def __str__(self):
        return f"{self.message} (Error code: {self.error_code})"

class IncompatibleFileFormatException(Exception):
    """Handles incompatible file format errors arising from pre-processing validation

    Args:
        Exception: Inheriting from the base Exception class
    """

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = 700
    
    def __str__(self):
        return f"{self.message} (Error code: {self.error_code})"

class EmptyFileExtensionException(Exception):
    """Handles files that do not have a suffix(extension), arising from pre-processing validation

    Args:
        Exception: Inheriting from the base Exception class
    """

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = 700.2
    
    def __str__(self):
        return f"{self.message} (Error code: {self.error_code})"