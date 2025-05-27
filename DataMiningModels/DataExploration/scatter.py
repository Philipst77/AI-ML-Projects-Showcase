import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df_original = pd.read_csv("Diabetes.csv")

df = df_original.copy()
pd.plotting.scatter_matrix(df, figsize=(7, 7));

plt.show()
