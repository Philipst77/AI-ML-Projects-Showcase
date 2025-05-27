import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

df2 = pd.read_csv("height_weight.csv")
m = df2["sex"] == 1

params = ["height", "weight"]
male_df = df2.loc[m, params].values
female_df = df2.loc[~m, params].values

c = ChainConsumer()
c.add_chain(male_df, parameters=params, name="Male", kde=1.0, color="b")
c.add_chain(female_df, parameters=params, name="Female", kde=1.0, color="r")

c.configure(
    contour_labels="confidence",
    usetex=False
)

c.plotter.plot(figsize=2.0)
plt.show()
