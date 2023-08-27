"""Extendable image data classes"""
import numpy as np


class Image(np.ndarray):
    """Generic image class that extends the np.ndarray class."""

    # From NumPy documentation
    # Add filename attribute
    def __new__(cls, input_array: np.ndarray, filename: str):
        obj = np.asarray(input_array).view(cls)
        # New attribute filename stores the path and filename of the source file
        obj.filename = filename
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.filename = getattr(obj, "filename", None)

    def __getitem__(self, key):
        # Enhance the np.ndarray __getitem__ method
        # Slice the array as requested but return an array of the same class
        # Idea from NumPy examples of subclassing:
        return super(Image, self).__getitem__(key)


class BGR(Image):
    """Subclass of Image for Blue, Green, Red (BGR) images."""

    def __new__(cls, input_array: np.ndarray, filename: str):
        return Image.__new__(cls, input_array, filename)

    def __getitem__(self, key):
        # Overwrite the __getitem__ method to return a GRAY object if the
        # requested slice is 2D
        new_arr = super(Image, self).__getitem__(key)
        if len(new_arr.shape) == 2:
            return GRAY(input_array=new_arr, filename=self.filename)
        return new_arr


class RGB(Image):
    """Subclass of Image for Red, Green, Blue (RGB) images."""

    def __new__(cls, input_array: np.ndarray, filename: str):
        return Image.__new__(cls, input_array, filename)

    def __getitem__(self, key):
        # Overwrite the __getitem__ method to return a GRAY object if the
        # requested slice is 2D
        new_arr = super(Image, self).__getitem__(key)
        if len(new_arr.shape) == 2:
            return GRAY(input_array=new_arr, filename=self.filename)
        return new_arr


class GRAY(Image):
    """Subclass of Image for grayscale images."""

    def __new__(cls, input_array: np.ndarray, filename: str):
        return Image.__new__(cls, input_array, filename)
