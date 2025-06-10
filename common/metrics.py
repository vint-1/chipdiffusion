import collections
import warnings

import numpy as np

class Metrics:

  def __init__(self):
    self.scalars = collections.defaultdict(list)
    self.collections = collections.defaultdict(list)
    self.aggs = {}
    self.lasts = {}

  def scalar(self, key, value, agg='mean'):
    assert agg in ('mean', 'sum', 'min', 'max')
    self.scalars[key].append(value)
    self.aggs[key] = agg

  def collection(self, key, value):
    # for 1D vector
    self.collections[key].append(value)

  def image(self, key, value):
    self.lasts[key] = value

  def video(self, key, value):
    self.lasts[key] = value

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        if len(value.shape) == 1:
          self.collection(key, value)
        if len(value.shape) > 1:
          self.lasts[key] = value
      else:
          self.scalar(key, value)

  def result(self, reset=True):
    result = {
        k: v for k, v in self.lasts.items()
    }
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self.scalars.items():
        agg = self.aggs[key]
        value = {
            'mean': np.nanmean,
            'sum': np.nansum,
            'min': np.nanmin,
            'max': np.nanmax,
        }[agg](values, dtype=np.float64)
        result[key] = value
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
