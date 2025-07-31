import jax.numpy as jnp
import jax
import numpy as np

def pad_images_to_square(images: jax.Array) -> jax.Array:
  """
  Pads a batch of JAX numpy images to make them square (1:1 aspect ratio).

  Args:
      images (jax.Array): Input images of shape (B, H, W, C), where B is the batch size.

  Returns:
      jax.Array: Padded images of shape (B, max(H, W), max(H, W), C).
  """
  is_single_image = False
  if images.ndim == 3:
    is_single_image = True
    images = images[None]
  B, H, W, C = images.shape
  assert images.ndim == 4, "Images must be a 4D array"
  max_dim = jnp.max(jnp.array([H, W]))  # Get the maximum dimension across the batch
  # Calculate padding for height and width
  pad_heights = (max_dim - H) // 2
  pad_widths = (max_dim - W) // 2
  # Create padding configuration for each image in the batch
  padding = ((0, 0), (pad_heights, max_dim - H - pad_heights), (pad_widths, max_dim - W - pad_widths), (0, 0))
  # Pad the images
  padded_images = np.pad(images, padding, mode='constant', constant_values=0)
  return padded_images[0] if is_single_image else padded_images

def xyxy2xywh(x: jax.Array) -> jax.Array:
  """
  Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
  top-left corner and (x2, y2) is the bottom-right corner.

  Args:
      x (jax.Array): The input bounding box coordinates in (x1, y1, x2, y2) format.

  Returns:
      jax.Array: The bounding box coordinates in (x, y, width, height) format.
  """
  assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
  # Calculate center coordinates
  x_center = (x[..., 0] + x[..., 2]) / 2  # x center
  y_center = (x[..., 1] + x[..., 3]) / 2  # y center
  # Calculate width and height
  width = x[..., 2] - x[..., 0]  # width
  height = x[..., 3] - x[..., 1]  # height
  # Stack the results
  return np.stack([x_center, y_center, width, height], axis=-1)


def xywh2xyxy(x: jax.Array) -> jax.Array:
  """
  Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
  top-left corner and (x2, y2) is the bottom-right corner.

  Args:
      x (jax.Array): The input bounding box coordinates in (x, y, width, height) format.

  Returns:
      jax.Array: The bounding box coordinates in (x1, y1, x2, y2) format.
  """
  assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
  # Extract centers and dimensions
  xy = x[..., :2]  # centers
  wh = x[..., 2:] / 2  # half width-height
  # Calculate top-left and bottom-right corners
  top_left = xy - wh  # top left xy
  bottom_right = xy + wh  # bottom right xy
  # Stack the results
  return np.concatenate([top_left, bottom_right], axis=-1)