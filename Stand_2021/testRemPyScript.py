def my_double(x):
    return x * 2

import pandas as pd

# define a python class
class explore:
  def __init__(self, df):
    self.df = df

  def metrics(self):
    desc = self.df.describe()
    return desc

  def dummify_vars(self):
    for field in self.df.columns:
        if isinstance(self.df[field][0], str):
          temp = pd.get_dummies(self.df[field])         
          self.df = pd.concat([self.df, temp], axis = 1)
          self.df.drop(columns = [field], inplace = True)
