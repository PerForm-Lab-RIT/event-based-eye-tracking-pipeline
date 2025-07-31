# %%

from abc import ABC, abstractmethod
from typing import Optional, Sequence
from functools import partial as bind
import numpy as np
import functools
import pathlib
import cv2
from PIL import Image
import os
from PIL import ImageOps
from PIL import ImageFilter
# from torchvision.transforms import transforms

class ImageProcessorMixin:

  """A mixin providing image processing and preprocessing functionalities."""

  def load_image(self, file_path):
    """Loads an image from the given path."""
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"File not found: {file_path}")
    return Image.open(file_path)

  def save_image(self, image, path, format="PNG"):
    """Saves an image to the specified path."""
    image.save(path, format=format)

  def resize_image(self, image, size):
    """Resizes the image to (width, height)."""
    return image.resize(size, Image.Resampling.LANCZOS)

  def center_crop_image_ratio(self, image, target_ratio: tuple):
    """
    Center crops the image to match the specified height/width ratio.

    target_ratio: float (width / height)

    Example:
    - target_ratio = 16/9  → Crop to a 16:9 aspect ratio
    - target_ratio = 1.0   → Crop to a square
    """
    width, height = image.size
    img_ratio = width / height  # Original aspect ratio
    _target_ratio = float(target_ratio[0]) / target_ratio[1]

    if img_ratio > _target_ratio:
      # Image is wider than target → crop width
      new_width = int(height * _target_ratio)
      new_height = height
    else:
      # Image is taller than target → crop height
      new_width = width
      new_height = int(width / _target_ratio)

    # Calculate cropping box (centered)
    left = (width - new_width) // 2
    upper = (height - new_height) // 2
    right = left + new_width
    lower = upper + new_height

    return image.crop((left, upper, right, lower))

  def center_crop(self, image, target_size: tuple):
    crop_width, crop_height = target_size
    width, height = image.size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

  def to_torch_format(self, image):
    """
    Converts a PIL image to a PyTorch tensor in (C, H, W) format.
    """
    img_array = np.array(image)  # Convert to NumPy (H, W, C)
    # if img_array.ndim == 2:  # Grayscale case
    #   img_array = img_array[:, :, None]  # Add channel dimension
    img_array = img_array.transpose(2, 0, 1)  # Convert (H, W, C) → (C, H, W)
    return img_array

  def _preprocess_image(self, image: Image.Image, **kwargs) -> Image.Image:
    """
    Applies a sequence of transformations based on the provided parameters.
    Expected kwargs:
    {
      "resize": (width, height),
      "crop": (16, 9),
      "rescale": scale_factor,
      "grayscale": True/False,
      "normalize": True/False,
      "torch_format": True/False
    }
    """
    # Make sure the loaded pil image is 3 dimensions, it can be 1 or three channels
    img_array = np.asarray(image)
    if img_array.ndim == 2:
      image = image.convert("L")

    if "resize" in kwargs:
      image = self.resize_image(image, kwargs["resize"])
    # if "crop" in kwargs:
    #   image = self.center_crop_image_ratio(image, kwargs["crop"])
    if "crop" in kwargs:
      image = self.center_crop(image, kwargs['crop'])
    if image.mode == "L":
      image = image.convert("RGB")
    elif image.mode == "RGBA":
      image = image.convert("RGB")
    elif image.mode == "RGB":
      pass
    else:
      raise ValueError(f"Image mode: {image.mode} is not supported")

    # Handle 2D image arrays (images without channel dimension)
    torch_format = kwargs.get("torch_format", False)
    if torch_format:
      image = self.to_torch_format(image)

    return image

  def preprocess_image(self, file_path_or_frame: str | pathlib.Path | np.ndarray, **kwargs) -> Image.Image:
    if isinstance(file_path_or_frame, (str, pathlib.Path)):
      img = self.load_image(file_path_or_frame)
    else:
      img = self.numpy_to_pil(file_path_or_frame)
    return self._preprocess_image(img, **kwargs)

  def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a numpy array.

    Args:
        image: PIL Image

    Returns:
        Numpy array with shape (H, W, C) for RGB or (H, W) for grayscale
    """
    return np.array(image)

  def numpy_to_pil(self, array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL Image.

    Args:
        array: Numpy array with shape (H, W, C) or (H, W)

    Returns:
        PIL Image
    """
    return Image.fromarray(array.astype(np.uint8))


class ImageAugmentationMixin:
  """A mixin providing various image augmentation techniques for data augmentation.

  All methods work with PIL Images and return PIL Images.
  """

  def random_horizontal_flip(self, image: Image.Image, p: float = 0.5) -> Image.Image:
    """Randomly flip an image horizontally with probability p.

    Args:
        image: PIL Image
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

  def random_vertical_flip(self, image: Image.Image, p: float = 0.5) -> Image.Image:
    """Randomly flip an image vertically with probability p.

    Args:
        image: PIL Image
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

  def random_rotation(self, image: Image.Image, degrees: float = 10, p: float = 0.5) -> Image.Image:
    """Randomly rotate an image by a degree within [-degrees, degrees].

    Args:
        image: PIL Image
        degrees: Maximum rotation angle in degrees
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      angle = np.random.uniform(-degrees, degrees)
      return image.rotate(angle, Image.Resampling.BILINEAR, expand=False)
    return image

  def color_jitter(self, image: Image.Image, brightness: float = 0.1, contrast: float = 0.1,
                  saturation: float = 0.1, hue: float = 0.05, p: float = 0.5) -> Image.Image:
    """Apply random color jittering to the image.

    Args:
        image: PIL Image
        brightness: Maximum brightness adjustment factor
        contrast: Maximum contrast adjustment factor
        saturation: Maximum saturation adjustment factor
        hue: Maximum hue adjustment factor
        p: Probability of applying each adjustment

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      # Convert PIL image to numpy array for easier manipulation
      img_array = np.array(image).astype(np.float32) / 255.0

      # Brightness adjustment
      if np.random.random() < p:
        factor = np.random.uniform(1-brightness, 1+brightness)
        img_array = img_array * factor
        img_array = np.clip(img_array, 0, 1)

      # Contrast adjustment
      if np.random.random() < p:
        factor = np.random.uniform(1-contrast, 1+contrast)
        mean = img_array.mean(axis=(0, 1), keepdims=True)
        img_array = (img_array - mean) * factor + mean
        img_array = np.clip(img_array, 0, 1)

      # Convert back to uint8 and PIL image for saturation and hue adjustments
      img = Image.fromarray((img_array * 255).astype(np.uint8))

      # Saturation adjustment
      if np.random.random() < p and image.mode == "RGB":
        factor = np.random.uniform(1-saturation, 1+saturation)
        img = Image.blend(img.convert("L").convert(image.mode), img, factor)

      # Hue adjustment
      if np.random.random() < p and image.mode == "RGB":
        img = img.convert("HSV")
        h, s, v = img.split()
        h_array = np.array(h)
        h_array = (h_array + np.random.uniform(-hue*180, hue*180)) % 180
        h = Image.fromarray(h_array.astype(np.uint8))
        img = Image.merge("HSV", (h, s, v)).convert("RGB")

      return img

    return image

  def gaussian_blur(self, image: Image.Image, radius_range: tuple = (0, 2), p: float = 0.5) -> Image.Image:
    """Apply Gaussian blur with random radius.

    Args:
        image: PIL Image
        radius_range: Range of radius values for the Gaussian kernel
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      radius = np.random.uniform(*radius_range)
      return image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image

  def random_noise(self, image: Image.Image, noise_level: float = 0.05, p: float = 0.5) -> Image.Image:
    """Add random Gaussian noise to an image.

    Args:
        image: PIL Image
        noise_level: Standard deviation of the Gaussian noise
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      img_array = np.array(image).astype(np.float32)
      noise = np.random.normal(0, noise_level * 255, img_array.shape)
      img_array = img_array + noise
      img_array = np.clip(img_array, 0, 255).astype(np.uint8)
      return Image.fromarray(img_array)
    return image

  def random_grayscale(self, image: Image.Image, p: float = 0.2) -> Image.Image:
    """Randomly convert image to grayscale with probability p.

    Args:
        image: PIL Image
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p and image.mode == "RGB":
      return image.convert("L").convert("RGB")
    return image

  def random_cutout(self, image: Image.Image, n_holes: int = 1, length: int = 50, p: float = 0.5) -> Image.Image:
    """Apply random cutout (erasing) on the image.

    Args:
        image: PIL Image
        n_holes: Number of holes to cut out
        length: Length of the square hole
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      img_array = np.array(image)
      h, w = img_array.shape[:2]

      for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        img_array[y1:y2, x1:x2] = 0

      return Image.fromarray(img_array)
    return image

  def random_affine(self, image: Image.Image, degrees: float = 10, translate: tuple = (0.1, 0.1), 
                   scale: tuple = (0.9, 1.1), shear: float = 10, p: float = 0.5) -> Image.Image:
    """Apply random affine transformation.

    Args:
        image: PIL Image
        degrees: Maximum rotation angle in degrees
        translate: Maximum translation in each direction as a fraction of width/height
        scale: Range of scaling factors
        shear: Maximum shear angle in degrees
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      width, height = image.size
      angle = np.random.uniform(-degrees, degrees)
      translations = (np.random.uniform(-translate[0], translate[0]) * width,
                     np.random.uniform(-translate[1], translate[1]) * height)
      scale_factor = np.random.uniform(scale[0], scale[1])
      shear_angle = np.random.uniform(-shear, shear)

      return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1/scale_factor, shear_angle/100, translations[0],
         shear_angle/100, 1/scale_factor, translations[1]),
        Image.Resampling.BILINEAR
      )
    return image

  def random_perspective(self, image: Image.Image, distortion_scale: float = 0.5, p: float = 0.5) -> Image.Image:
    """Apply random perspective transformation.

    Args:
        image: PIL Image
        distortion_scale: Scale of distortion, higher means more distortion
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      width, height = image.size

      # Get random perspective transformation coefficients
      half_width = width // 2
      half_height = height // 2
      topleft = (np.random.uniform(0, distortion_scale) * half_width,
                np.random.uniform(0, distortion_scale) * half_height)
      topright = (width - np.random.uniform(0, distortion_scale) * half_width,
                 np.random.uniform(0, distortion_scale) * half_height)
      botright = (width - np.random.uniform(0, distortion_scale) * half_width,
                 height - np.random.uniform(0, distortion_scale) * half_height)
      botleft = (np.random.uniform(0, distortion_scale) * half_width,
                height - np.random.uniform(0, distortion_scale) * half_height)

      coeffs = self._get_perspective_coeffs(
        [0, 0, width, 0, width, height, 0, height],
        [topleft[0], topleft[1], topright[0], topright[1],
         botright[0], botright[1], botleft[0], botleft[1]]
      )

      return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC
      )
    return image

  def _get_perspective_coeffs(self, source_coords, target_coords):
    """Calculate coefficients for perspective transformation."""
    matrix = []
    for i in range(0, 8, 2):
      x, y = source_coords[i:i+2]
      X, Y = target_coords[i:i+2]
      matrix.extend([x, y, 1, 0, 0, 0, -X*x, -X*y])
      matrix.extend([0, 0, 0, x, y, 1, -Y*x, -Y*y])

    A = np.matrix(matrix, dtype=np.float32).reshape(8, 8)
    B = np.array(target_coords, dtype=np.float32)

    # Solve the system
    res = np.linalg.solve(A, B)
    return tuple(res.flatten().tolist())

  def channel_shuffle(self, image: Image.Image, p: float = 0.2) -> Image.Image:
    """Randomly shuffle the channels in an RGB image.

    Args:
        image: PIL Image
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p and image.mode == "RGB":
      r, g, b = image.split()
      channels = [r, g, b]
      np.random.shuffle(channels)
      return Image.merge("RGB", channels)
    return image

  def salt_and_pepper_noise(self, image: Image.Image, prob: float = 0.01, p: float = 0.2) -> Image.Image:
    """Add salt and pepper noise to the image.

    Args:
        image: PIL Image
        prob: Probability of changing a pixel to salt or pepper
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      img_array = np.array(image)
      height, width = img_array.shape[:2]

      # Salt (white) noise
      num_salt = int(prob * height * width)
      salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
      if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[2]):
          img_array[salt_coords[0], salt_coords[1], i] = 255
      else:  # Grayscale
        img_array[salt_coords[0], salt_coords[1]] = 255

      # Pepper (black) noise
      num_pepper = int(prob * height * width)
      pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
      if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[2]):
          img_array[pepper_coords[0], pepper_coords[1], i] = 0
      else:  # Grayscale
        img_array[pepper_coords[0], pepper_coords[1]] = 0

      return Image.fromarray(img_array)
    return image

  def adjust_gamma(self, image: Image.Image, gamma_range: tuple = (0.8, 1.2), p: float = 0.5) -> Image.Image:
    """Apply gamma correction with random gamma value.

    Args:
        image: PIL Image
        gamma_range: Range of gamma values
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      gamma = np.random.uniform(*gamma_range)
      img_array = np.array(image).astype(np.float32) / 255.0
      img_array = np.power(img_array, gamma)
      img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
      return Image.fromarray(img_array)
    return image

  def random_crop_resize(self, image: Image.Image, scale: tuple = (0.8, 1.0), 
                       ratio: tuple = (0.75, 1.33), p: float = 0.5) -> Image.Image:
    """Randomly crop and resize the image.

    Args:
        image: PIL Image
        scale: Range of scale for the cropped area relative to original image
        ratio: Range of aspect ratio for the cropped area
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      width, height = image.size
      area = width * height

      for _ in range(10):  # Try 10 times to find a suitable random crop
        target_area = np.random.uniform(*scale) * area
        log_ratio = np.log(ratio)
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))

        w = int(np.sqrt(target_area * aspect_ratio))
        h = int(np.sqrt(target_area / aspect_ratio))

        if 0 < w <= width and 0 < h <= height:
          x = np.random.randint(0, width - w + 1)
          y = np.random.randint(0, height - h + 1)

          cropped = image.crop((x, y, x + w, y + h))
          return cropped.resize((width, height), Image.Resampling.BILINEAR)

    return image

  def solarize(self, image: Image.Image, threshold: int = 128, p: float = 0.2) -> Image.Image:
    """Invert all pixel values above a threshold.

    Args:
        image: PIL Image
        threshold: All pixels above this value will be inverted
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      return Image.fromarray(np.where(
        np.array(image) >= threshold,
        255 - np.array(image),
        np.array(image)
      ).astype(np.uint8))
    return image

  def posterize(self, image: Image.Image, bits: int = 4, p: float = 0.2) -> Image.Image:
    """Reduce the number of bits for each color channel.

    Args:
        image: PIL Image
        bits: Number of bits to keep for each channel (1-8)
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      img_array = np.array(image)
      img_array = (img_array // (2**(8-bits))) * (2**(8-bits))
      return Image.fromarray(img_array.astype(np.uint8))
    return image

  def equalize_histogram(self, image: Image.Image, p: float = 0.2) -> Image.Image:
    """Equalize the histogram of the image.

    Args:
        image: PIL Image
        p: Probability of applying the transformation

    Returns:
        Augmented PIL Image
    """
    if np.random.random() < p:
      if image.mode == "L":
        return ImageOps.equalize(image)
      else:  # Color image
        r, g, b = image.split()
        r_eq = ImageOps.equalize(r)
        g_eq = ImageOps.equalize(g)
        b_eq = ImageOps.equalize(b)
        return Image.merge("RGB", (r_eq, g_eq, b_eq))
    return image

  def apply_augmentations(self, image: Image.Image, augmentations: list | None = None) -> Image.Image:
    """
    Apply a series of augmentations to an image.

    Args:
        image: PIL Image to augment
        augmentations: List of dictionaries with augmentation settings
                      [{"name": "random_horizontal_flip", "params": {"p": 0.5}}, ...]
                      If None, apply default augmentations

    Returns:
        Augmented PIL Image
    """
    if augmentations is None:
      # Default augmentation pipeline with reasonable defaults
      augmentations = [
        {"name": "random_horizontal_flip", "params": {"p": 0.5}},
        {"name": "color_jitter", "params": {"brightness": 0.1, "contrast": 0.1,
                                           "saturation": 0.1, "hue": 0.05, "p": 0.2}},
        {"name": "gaussian_blur", "params": {"radius_range": (0, 1), "p": 0.1}},
        {"name": "random_grayscale", "params": {"p": 0.1}},
        {"name": "salt_and_pepper_noise", "params": {"prob": 0.0005, "p": 0.1}},
        {"name": "random_crop_resize", "params": {"scale": (0.8, 1.0), "ratio": (0.75, 1.33), "p": 0.5}},
        {"name": "posterize", "params": {"bits": 4, "p": 0.1}},
        {"name": "equalize_histogram", "params": {"p": 0.05}},
      ]

    # Apply each augmentation in sequence
    img = image.copy()
    for aug in augmentations:
      method = getattr(self, aug["name"])
      img = method(img, **aug.get("params", {}))

    return img


if __name__ == '__main__':
  a = ImageProcessorMixin()
  img = a.preprocess_image("resources/a.jpg", resize=(256,256), crop=(256, 256))

  # Test augmentations
  aug = ImageAugmentationMixin()
  augmented = aug.apply_augmentations(img)
  import matplotlib.pyplot as plt
  plt.imshow(augmented)
  plt.show()

