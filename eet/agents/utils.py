
import jax
import jax.numpy as jnp
import numpy as np

# Color for colorblind-friendly plots
BLUE = (0, 114, 178)
ORANGE = (230, 159, 0)
GREEN = (0, 158, 115)
YELLOW = (240, 228, 66)
PURPLE = (204, 121, 167)
GRAY = (128, 128, 128)
RED = (213, 94, 0)

CROSS_HAIR_THICKNESS = 2
CROSS_HAIR_LENGTH = 5

ispupilcentroid = lambda s: np.issubdtype(s.dtype, np.integer) and s.shape == (2,)
# normpupilcentroid = lambda x, h: x / h * 2 - 1
# unnormpupilcentroid = lambda x, h: (x + 1) / 2 * h
normpupilcentroid = lambda x, h: x - h / 2.0 # maybe we subtract half of it but not divide
unnormpupilcentroid = lambda x, h: x + h / 2.0
isimage = lambda s: len(s.shape) == 3

def draw_dot(image: jax.Array, cy: jax.Array, cx: jax.Array, r: int, color: tuple = (0, 0, 0)):
  """_summary_

  Args:
      image (jax.Array): actual image in batch in uint8 (..., H, W, C)
      cy (jax.Array): center y. (...,)
      cx (jax.Array): center x. (...,)
      r (int): radius of the dot
      color (tuple, optional): color of the dot. Defaults to (0, 0, 0).

  Returns:
      _type_: _description_
  """
  assert image.dtype == jnp.uint8 and image.shape[-1] == 3, (image.dtype, image.shape)
  *bdims, H, W, C = image.shape
  bshape = tuple(bdims)
  # Create meshgrid for spatial dimensions
  y = jnp.arange(H)
  x = jnp.arange(W)
  yy, xx = jnp.meshgrid(y, x, indexing='ij')  # (H, W)
  # Expand and broadcast to match batch dimensions
  yy = yy.reshape((1,) * len(bshape) + (H, W))
  xx = xx.reshape((1,) * len(bshape) + (H, W))
  # Reshape cy, cx to align with image dimensions
  cy = cy.reshape(bshape + (1, 1))
  cx = cx.reshape(bshape + (1, 1))
  # Compute squared distances
  dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2  # shape: (*batch_dims, H, W)
  mask = dist_sq <= r ** 2  # shape: (*batch_dims, H, W)
  mask = mask[..., None]  # shape: (*batch_dims, H, W, 1)
  # Reshape color for broadcasting
  color = jnp.array(color, dtype=image.dtype)
  color = color.reshape((1,) * len(bshape) + (1, 1, 3))
  # Draw dot
  return jnp.where(mask, color, image)


def draw_cross_hair(
    image: jax.Array,
    cy: jax.Array, cx: jax.Array,
    length: int = CROSS_HAIR_LENGTH,
    thickness: int = CROSS_HAIR_THICKNESS,
    color: tuple = (0, 0, 0)
):
  """Draw a cross hair (horizontal and vertical lines) on an image.

  Args:
      image (jax.Array): actual image in batch in uint8 (..., H, W, C)
      cy (jax.Array): center y. (...,)
      cx (jax.Array): center x. (...,)
      length (int): half-length of each line from the center
      thickness (int, optional): thickness of the lines. Defaults to 1.
      color (tuple, optional): color of the cross hair. Defaults to (0, 0, 0).

  Returns:
      jax.Array: image with cross hair drawn
  """
  assert image.dtype == jnp.uint8 and image.shape[-1] == 3, (image.dtype, image.shape)
  *bdims, H, W, C = image.shape
  bshape = tuple(bdims)

  # Create meshgrid for spatial dimensions
  y = jnp.arange(H)
  x = jnp.arange(W)
  yy, xx = jnp.meshgrid(y, x, indexing='ij')  # (H, W)

  # Expand and broadcast to match batch dimensions
  yy = yy.reshape((1,) * len(bshape) + (H, W))
  xx = xx.reshape((1,) * len(bshape) + (H, W))

  # Reshape cy, cx to align with image dimensions
  cy = cy.reshape(bshape + (1, 1))
  cx = cx.reshape(bshape + (1, 1))

  # Create diagonal line mask (top-left to bottom-right)
  # Line equation: y - cy = x - cx, or |y - cy - (x - cx)| <= thickness
  diagonal1_mask = (
    (jnp.abs((yy - cy) - (xx - cx)) <= thickness // 2) &  # on diagonal line
    (jnp.abs(xx - cx) <= length) &                        # within length horizontally
    (jnp.abs(yy - cy) <= length)                          # within length vertically
  )

  # Create diagonal line mask (top-right to bottom-left)
  # Line equation: y - cy = -(x - cx), or |y - cy + (x - cx)| <= thickness
  diagonal2_mask = (
    (jnp.abs((yy - cy) + (xx - cx)) <= thickness // 2) &  # on diagonal line
    (jnp.abs(xx - cx) <= length) &                        # within length horizontally
    (jnp.abs(yy - cy) <= length)                          # within length vertically
  )

  # Combine masks (union of both diagonal lines)
  cross_mask = diagonal1_mask | diagonal2_mask  # shape: (*batch_dims, H, W)
  cross_mask = cross_mask[..., None]  # shape: (*batch_dims, H, W, 1)

  # Reshape color for broadcasting
  color = jnp.array(color, dtype=image.dtype)
  color = color.reshape((1,) * len(bshape) + (1, 1, 3))

  # Draw cross hair
  return jnp.where(cross_mask, color, image)


def draw_cross_hair_with_dot(
    image: jax.Array,
    cy: jax.Array, cx: jax.Array,
    length: int = CROSS_HAIR_LENGTH,
    thickness: int = CROSS_HAIR_THICKNESS,
    gap: int = 2,
    dot_radius: int = 1,
    color: tuple = (0, 0, 0)
):
  """Draw an X-shaped cross hair with a gap toward center and a dot in the middle.

  Args:
      image (jax.Array): actual image in batch in uint8 (..., H, W, C)
      cy (jax.Array): center y. (...,)
      cx (jax.Array): center x. (...,)
      length (int): half-length of each line from the center
      thickness (int, optional): thickness of the lines. Defaults to CROSS_HAIR_THICKNESS.
      gap (int, optional): gap from center where lines don't draw. Defaults to 2.
      dot_radius (int, optional): radius of center dot. Defaults to 1.
      color (tuple, optional): color of the cross hair and dot. Defaults to (0, 0, 0).

  Returns:
      jax.Array: image with cross hair and center dot drawn
  """
  assert image.dtype == jnp.uint8 and image.shape[-1] == 3, (image.dtype, image.shape)
  *bdims, H, W, C = image.shape
  bshape = tuple(bdims)

  # Create meshgrid for spatial dimensions
  y = jnp.arange(H)
  x = jnp.arange(W)
  yy, xx = jnp.meshgrid(y, x, indexing='ij')  # (H, W)

  # Expand and broadcast to match batch dimensions
  yy = yy.reshape((1,) * len(bshape) + (H, W))
  xx = xx.reshape((1,) * len(bshape) + (H, W))

  # Reshape cy, cx to align with image dimensions
  cy = cy.reshape(bshape + (1, 1))
  cx = cx.reshape(bshape + (1, 1))

  # Create diagonal line mask (top-left to bottom-right)
  # Line equation: y - cy = x - cx, or |y - cy - (x - cx)| <= thickness
  diagonal1_mask = (
    (jnp.abs((yy - cy) - (xx - cx)) <= thickness // 2) &  # on diagonal line
    (jnp.abs(xx - cx) <= length) &                        # within length horizontally
    (jnp.abs(yy - cy) <= length) &                        # within length vertically
    ((jnp.abs(xx - cx) >= gap) | (jnp.abs(yy - cy) >= gap))  # outside gap zone
  )

  # Create diagonal line mask (top-right to bottom-left)
  # Line equation: y - cy = -(x - cx), or |y - cy + (x - cx)| <= thickness
  diagonal2_mask = (
    (jnp.abs((yy - cy) + (xx - cx)) <= thickness // 2) &  # on diagonal line
    (jnp.abs(xx - cx) <= length) &                        # within length horizontally
    (jnp.abs(yy - cy) <= length) &                        # within length vertically
    ((jnp.abs(xx - cx) >= gap) | (jnp.abs(yy - cy) >= gap))  # outside gap zone
  )

  # Combine diagonal masks
  cross_mask = diagonal1_mask | diagonal2_mask  # shape: (*batch_dims, H, W)

  # Create center dot mask
  dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2  # shape: (*batch_dims, H, W)
  dot_mask = dist_sq <= dot_radius ** 2  # shape: (*batch_dims, H, W)

  # Combine cross hair and dot masks
  combined_mask = cross_mask | dot_mask  # shape: (*batch_dims, H, W)
  combined_mask = combined_mask[..., None]  # shape: (*batch_dims, H, W, 1)

  # Reshape color for broadcasting
  color = jnp.array(color, dtype=image.dtype)
  color = color.reshape((1,) * len(bshape) + (1, 1, 3))

  # Draw cross hair with dot
  return jnp.where(combined_mask, color, image)



