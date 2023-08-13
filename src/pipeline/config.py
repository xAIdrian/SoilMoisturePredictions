import sys
import os
import matplotlib.pyplot as plt

def set_config():
  # Get the absolute path of the root directory (adjust according to your specific structure)
  # Add the root path to sys.path
  root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  sys.path.append(root_path)

  plt.style.use('fivethirtyeight')
  plt.rcParams['figure.figsize'] = (20, 5)
  plt.rcParams['figure.dpi'] = 100
  plt.rcParams['lines.linewidth'] = 2
