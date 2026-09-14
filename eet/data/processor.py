from abc import ABC, abstractmethod
from transformers import ProcessorMixin

from .types import BatchItemDataType, EyeEpisodeDataType
from ..utils import Space


class BaseProcessor(ProcessorMixin, ABC):
  """Base class for event batch item data processing.

    This class provides a common interface and shared functionality for processing
    event batch items in the eye tracking pipeline.
  """
  def __call__(self, item: BatchItemDataType) -> None:
    """Process a single batch item. Put the processed data into the processed frame

    Args:
        item (BatchItemDataType): The batch item to process, containing raw episode data and metadata.
    """
    pass

  @property
  @abstractmethod
  def input_space(self) -> Space:
    """
    Return the input space of the dataset, which defines the shape and type of the input data.

    Returns:
      Space object representing the input space
    """
    pass

  @property
  @abstractmethod
  def label_space(self) -> Space:
    """
    Return the label space of the dataset, which defines the shape and type of the labels.

    Returns:
      Space object representing the label space
    """
    pass


