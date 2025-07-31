import numpy as np

def confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> tuple:
  """Compute the confusion matrix from the labels and predictions.
    Because this is a softmax outputs, we need to 

  Args:
      labels (np.ndarray): The labels: (..., num_classes) or (...,), binary value: 0, 1
      preds (np.ndarray): The predictions: (..., num_classes). Binary value: 0, 1

  Returns:
      tuple: The confusion matrix: TP, FP, TN, FN
  """
  if labels.ndim < preds.ndim:
    num_classes = preds.shape[-1]
    labels = np.eye(num_classes)[labels] # (..., num_classes)
  assert labels.shape == preds.shape, f"labels and preds must have the same shape. Got labels: {labels.shape} and preds: {preds.shape}"
  # NOTE: instead of -1, we put 0
  # True positive: all positions in preds and labels are positive predictions, then take the matches
  TP = np.logical_and(labels == 1, preds == 1).astype(np.float32).sum() # ()
  # False positive: all positions in preds and labels are positive predictions, then take the unmatches
  FP = np.logical_and(labels == 0, preds == 1).astype(np.float32).sum() # ()
  # True negative: all positions in preds and labels are negative predictions, then take the matches
  TN = np.logical_and(labels == 0, preds == 0).astype(np.float32).sum() # ()
  # False negative: all positions in preds and labels are negative predictions, then take the unmatches
  FN = np.logical_and(labels == 1, preds == 0).astype(np.float32).sum() # ()
  return TP, FP, TN, FN

def confusion_matrix_with_normalization(labels: np.ndarray, preds: np.ndarray) -> tuple:
  """Compute the confusion matrix from the labels and predictions.

  Args:
      labels (np.ndarray): The labels: (..., num_classes) or (...,), binary value: 0, 1
      preds (np.ndarray): The predictions: (..., num_classes). Binary value: 0, 1

  Returns:
      tuple: The confusion matrix: TP, FP, TN, FN
  """
  assert np.any((preds == 0) | (preds == 1)), f"preds must be binary values: 0, 1. Got {preds}"
  assert np.any((labels == 0) | (labels == 1)), f"labels must be binary values: 0, 1. Got {labels}"
  if labels.ndim < preds.ndim:
    num_classes = preds.shape[-1]
    labels = np.eye(num_classes)[labels] # (..., num_classes)
  # if label == 1 and pred == 1, then it is a correct prediction, we normalize it
  correct = np.where((preds == 1) & (labels == 1), 1, 0) # all zeros if they are not correct
  spressed = correct + preds
  spressed = np.where(np.any(spressed > 1, axis=1, keepdims=True), np.where(spressed == 1, 0, spressed), spressed)
  spressed = np.where(spressed > 0, 1, 0)
  # spressed = np.where((preds == 1) & (labels == 1), 1, preds)
  # print(f"spressed: {spressed}")
  _preds = np.argmax(spressed, axis=-1) # (B,)
  # print(f"_preds: {_preds}")
  _labels = np.argmax(labels, axis=-1) # (B,)
  # print(f"_labels: {_labels}")
  flat_index = num_classes * _labels + _preds
  # print(f"flat_index: {flat_index}")
  confusion_matrix = np.bincount(flat_index,
    minlength=num_classes*num_classes).reshape(num_classes, num_classes)
  # Compute TP, FP, FN, TN
  TP = np.diag(confusion_matrix)  # True positives are the diagonal elements
  FP = np.sum(confusion_matrix, axis=0) - TP  # False positives are column sums minus diagonal
  FN = np.sum(confusion_matrix, axis=1) - TP  # False negatives are row sums minus diagonal
  TN = np.sum(confusion_matrix) - (TP + FP + FN)  # Remaining elements are true negatives
  return confusion_matrix, TP, FP, TN, FN

