
from transformers import ProcessorMixin

from .types import BatchItemDataType, EyeEpisodeDataType

class BaseProcessor(ProcessorMixin):
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

