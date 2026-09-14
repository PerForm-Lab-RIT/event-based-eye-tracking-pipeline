

# Collator takes data and put them into the batch

import torch
from typing import List, Dict, Any
from .types import BatchItemDataType

class GeneralCollator:
  """
  We just stack the processed field of every batch item together
  """
  # def __call__(self, items: List[BatchItemDataType]) -> Dict[str, Any]:
  #   all_items = [b.processed for b in items]
  #   batch = {k: torch.stack([item[k] for item in all_items], dim=0) for k in all_items[0].keys()}
  #   return batch
  pass
