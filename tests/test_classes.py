import numpy as np
from edic import Image, BGR, RGB, GRAY


def test_image():
    """Test creating an Image class image."""
    img = Image(input_array=np.zeros((10, 10), dtype=np.uint8), filename="image.png")
    assert isinstance(img, Image)


def test_image_none():
    """Test creating an Image class image."""
    img = Image(input_array=None, filename=None)
    assert isinstance(img, Image)


def test_image_slice():
    """Test subsetting an Image."""
    img = Image(input_array=np.zeros((10, 10), dtype=np.uint8), filename="image.png")
    assert img[0:5, 0:5].shape == (5, 5)


def test_bgr():
    """Test creating a BGR class image."""
    bgr = BGR(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="bgr.png")
    assert isinstance(bgr, BGR)


def test_bgr_slice():
    """Test subsetting a BGR image."""
    bgr = BGR(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="bgr.png")
    assert bgr[0:5, 0:5].shape == (5, 5, 3)


def test_rgb():
    """Test creating a RGB class image."""
    bgr = RGB(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="rgb.png")
    assert isinstance(bgr, RGB)


def test_rgb_slice():
    """Test subsetting an RGB image."""
    rgb = RGB(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="rgb.png")
    assert rgb[0:5, 0:5].shape == (5, 5, 3)


def test_gray():
    """Test creating a GRAY class image."""
    gray = GRAY(input_array=np.zeros((10, 10), dtype=np.uint8), filename="gray.png")
    assert isinstance(gray, GRAY)


def test_bgr_to_gray():
    """Test converting a BGR image to a GRAY image."""
    bgr = BGR(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="bgr.png")
    gray = bgr[:, :, 0]
    assert isinstance(gray, GRAY)


def test_rgb_to_gray():
    """Test converting a RGB image to a GRAY image."""
    rgb = RGB(input_array=np.zeros((10, 10, 3), dtype=np.uint8), filename="rgb.png")
    gray = rgb[:, :, 0]
    assert isinstance(gray, GRAY)
