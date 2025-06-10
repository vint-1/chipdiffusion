import collections
import warnings

import numpy as np

class Metrics:

  agg_fns = {
    'mean': np.nanmean,
    'sum': np.nansum,
    'min': np.nanmin,
    'max': np.nanmax,
    'std': np.nanstd,
    'median': np.nanmedian,
  }

  def __init__(self):
    self.scalars = collections.defaultdict(list)
    self.collections = collections.defaultdict(list)
    self.aggs = {}
    self.lasts = {}

  def scalar(self, key, value, aggs=('mean', 'std', 'min', 'max')):
    assert all([agg in Metrics.agg_fns for agg in aggs])
    self.scalars[key].append(value)
    self.aggs[key] = aggs

  def collection(self, key, value):
    # for 1D vector
    self.collections[key].append(value)

  def image(self, key, value):
    self.lasts[key] = value

  def video(self, key, value):
    self.lasts[key] = value

  def add(self, mapping, aggs_map={}, prefix=None):
    for key, value in mapping.items():
      prefixed_key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        if len(value.shape) == 1:
          self.collection(prefixed_key, value)
        if len(value.shape) > 1:
          self.lasts[prefixed_key] = value
      else:
          if key in aggs_map:
            self.scalar(prefixed_key, value, aggs=aggs_map[key])
          else:
            self.scalar(prefixed_key, value)

  def result(self, reset=True):
    result = {
        k: v for k, v in self.lasts.items()
    }
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self.scalars.items():
        for agg in self.aggs[key]:
          value = Metrics.agg_fns[agg](values)
          newkey = f"{key}_{agg}" if len(self.aggs[key])>1 else key
          result[newkey] = value 
      for key, values in self.collections.items():
        result[key] = np.concatenate(values, axis=0)
    reset and self.reset()
    return result
  
  def get_all_as_collections(self):
    result = {}
    for key, values in self.collections.items():
        result[key] = np.concatenate(values, axis=0)
    for key, values in self.scalars.items():
        value = np.array(values)
        result[key] = value
    return result

  def reset(self):
    self.scalars.clear()
    self.collections.clear()
    self.lasts.clear()