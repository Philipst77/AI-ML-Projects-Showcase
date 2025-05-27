import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df_original = pd.read_csv("Diabetes.csv")
df = df_original.copy()
df.corr()
sb.heatmap(df.corr());
sb.heatmap(df.corr(), annot=True, cmap="viridis", fmt="0.2f");
plt.show()
