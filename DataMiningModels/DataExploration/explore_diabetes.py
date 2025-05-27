#🧠 These lines import libraries you’ll need:
#numpy (np): for numerical stuff (like NaN)

#pandas (pd): for handling datasets (tables)

#matplotlib.pyplot (plt): for plotting graphs

#seaborn (sb): prettier statistical plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#📂 Loads the dataset
#Reads the Diabetes.csv file into a table (DataFrame) called df_original.
df_original = pd.read_csv("Diabetes.csv")

df_original.head()
# This just prints a preview so you can see what the data looks like.
cols = [c for c in df_original.columns if c not in ["Pregnancies", "Outcome"]]
#🧼 Gets all columns except:
#Pregnancies (which you’re keeping as-is)
#Outcome (target column: 0 = no diabetes, 1 = yes diabetes)
#This is because the next step will clean the other columns only.
#
df = df_original.copy()
df[cols] = df[cols].replace({0: np.nan})
#🧽 Cleans up the data
#Makes a copy of the original data
#Replaces any 0 values in Glucose, BloodPressure, etc., with np.nan (missing)
#Because 0 isn’t valid for those medical features
df.head()
df.info()
df.describe()

#.head(): shows the first 5 rows
#.info(): shows how many missing values you now have
#.describe(): gives mean, std, min, max of each column


#You're cleaning and understanding the data:

#Remove invalid 0s → replaced with missing values

#Understand how many missing values you have

#Know the spread of your data (mean, max, std dev, etc.)

#This step prepares your data so you can:

#Plot it (scatter, histograms, etc.)

#Run correlation checks

#Feed it into machine learning models


