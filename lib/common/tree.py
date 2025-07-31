
from typing import List, Dict, Tuple, Union, Callable, Any
import fnmatch
from . import printing


def map_(fn, *trees, isleaf=None):
  assert trees, 'Provide one or more nested Python structures'
  kw = dict(isleaf=isleaf)
  first = trees[0]
  assert all(isinstance(x, type(first)) for x in trees)
  if isleaf and isleaf(first):
    return fn(*trees)
  if isinstance(first, list):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return [map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
  if isinstance(first, tuple):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return tuple([map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
  if isinstance(first, dict):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return {k: map_(fn, *[t[k] for t in trees], **kw) for k in first}
  if hasattr(first, 'keys') and hasattr(first, 'get'):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return type(first)(
        {k: map_(fn, *[t[k] for t in trees], **kw) for k in first})
  return fn(*trees)

def leaves_(tree, is_leaf=None):
  kw = dict(is_leaf=is_leaf)
  result = []
  if is_leaf and is_leaf(tree):
    result.append(tree)
    return result
  if isinstance(tree, list):
    for t in tree:
      li = leaves_(t, **kw)
      [result.append(item) for item in li]
  elif isinstance(tree, tuple):
   for t in tree:
      li = leaves_(t, **kw)
      [result.append(item) for item in li]
  elif isinstance(tree, dict):
    for k, v in tree.items():
      li = leaves_(v, **kw)
      [result.append(item) for item in li]
  elif hasattr(tree, 'keys') and hasattr(tree, 'get'):
    for k in tree:
      li = leaves_(tree[k], **kw)
      [result.append(item) for item in li]
  else:
    result.append(tree)
  return result

def selective_map_(
    fn: Callable,
    match: Union[str, Callable[[str, Any], bool]],
    tree: Dict[str, Any],
    *,
    _keypath: str = "",
) -> Dict[str, Any]:
    """Maps a function over a nested dictionary, only applying it leaves that match a criterion.

    If `match` is a string, it follows glob-style syntax. For example, "bar" will only match
    a top-level key called "bar", "*bar" will match any leaf whose key ends with "bar",
    and "*bar*" will match any subtree with a key that contains "bar".

    Key paths are separated by "/". For example, "foo/bar" will match a leaf with key "bar" that
    is nested under a key "foo".

    Args:
      fn (Callable): The function to apply.
      match (str or Callable[[str, Any], bool]): If a string or list of strings, `map_fn` will
          only be applied to leaves whose key path matches `match` using glob-style syntax. If a
          function, `map_fn` will only be applied to leaves for which `match(key_path, value)`
          returns True.
      tree (Dict[str, Any]): The (possibly nested) dictionary to map over.
    """
    if not callable(match):
      match_fn = lambda keypath, value: fnmatch.fnmatch(keypath, match)
    else:
      match_fn = match

    out = {}
    for key in tree:
      if isinstance(tree[key], dict):
        out[key] = selective_map_(fn, match_fn, tree[key], _keypath=_keypath + key + "/")
      elif match_fn(_keypath + key, tree[key]):
        out[key] = fn(tree[key])
      else:
        out[key] = tree[key]
    return out


# def map(fn, *trees, isleaf=None):
#   assert trees, 'Provide one or more nested Python structures'
#   kw = dict(isleaf=isleaf)
#   first = trees[0]
#   try:
#     assert all(isinstance(x, type(first)) for x in trees)
#     if isleaf and isleaf(trees[0]):
#       return fn(*trees)
#     if isinstance(first, list):
#       assert all(len(x) == len(first) for x in trees)
#       return [map(
#           fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
#     if isinstance(first, tuple):
#       assert all(len(x) == len(first) for x in trees)
#       return tuple([map(
#           fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
#     if isinstance(first, dict):
#       assert all(set(x.keys()) == set(first.keys()) for x in trees)
#       return {k: map(fn, *[t[k] for t in trees], **kw) for k in first}
#     if hasattr(first, 'keys') and hasattr(first, 'get'):
#       assert all(set(x.keys()) == set(first.keys()) for x in trees)
#       return type(first)(
#           {k: map(fn, *[t[k] for t in trees], **kw) for k in first})
#   except AssertionError:
#     raise TypeError(printing.format_(trees))
#   return fn(*trees)


def flatten(tree, isleaf=None):
  leaves = []
  map(lambda x: leaves.append(x), tree, isleaf=isleaf)
  structure = map(lambda x: None, tree, isleaf=isleaf)
  return tuple(leaves), structure


def unflatten(leaves, structure):
  leaves = iter(tuple(leaves))
  return map(lambda x: next(leaves), structure)


def flatdict(tree, sep='/'):
  assert isinstance(tree, (dict, tuple)), type(tree)
  mapping = {}
  for key, value in tree.items():
    if isinstance(value, dict):
      inner = flatdict(value)
      mapping.update({f'{key}{sep}{k}': v for k, v in inner.items()})
    elif isinstance(value, tuple):
      inner = flatdict({f'[{i}]': x for i, x in enumerate(value)})
      mapping.update({f'{key}{sep}{k}': v for k, v in inner.items()})
    else:
      mapping[key] = value
  return mapping


def nestdict(mapping, sep='/'):
  assert isinstance(mapping, dict)
  tree = {}
  for path, value in mapping.items():
    node = tree
    parts = path.split(sep)
    for part in parts[:-1]:
      node = node.setdefault(part, {})
    node[parts[-1]] = value
  def post(tree):
    if isinstance(tree, dict):
      tree = {k: post(v) for k, v in tree.items()}
      if all(k.startswith('[') and k.endswith(']') for k in tree):
        available = set(int(x[1:-1]) for x in tree.keys())
        assert available == set(range(len(tree))), available
        tree = tuple(tree[f'[{i}]'] for i in range(len(tree)))
    return tree
  tree = post(tree)
  return tree



leaves = leaves_
map = map_
selective_map = selective_map_
