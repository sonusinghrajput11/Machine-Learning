import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns

#Reading boston dataset
boston_dataset = load_boston()

#Reading data part
features = boston_dataset.data

#Reading features name
features_name = boston_dataset.feature_names

#Creating Data Frames
dataset = pd.DataFrame(data = features, columns = features_name)

#Calculation of correlation coefficient
corr_coff = dataset.corr()

#Visualising correlation coefficient using Seaborn
sns.heatmap(corr_coff, xticklabels=corr_coff.columns.values, yticklabels=corr_coff.columns.values)

#Visualising correlation coefficient using panda only
corr_coff.style.background_gradient()

#############################Finding Highly correlated features#####################################
    
#Step 1. Remove self correlations - One way of doing that by setting diagonal values as Zero
np.fill_diagonal(corr_coff.values, 0)
    
#Step 2. Unstack coefficients matrix 
unstacked_coff = corr_coff.abs().unstack()

#Step 3. Sort the unstacked coefficients matrix
soretd_unstacked_coff = unstacked_coff.sort_values(kind = "quicksort", ascending=False)

#Step 4. Drop duplicates
soretd_unstacked_coff_final = soretd_unstacked_coff.drop_duplicates()
soretd_unstacked_coff_final

#############################Visualising Mean, Variance and Std deviation#############################

#Finding Mean
ds_mean = dataset.mean()

#Finding Variance
ds_variance = dataset.var()

#Finding std deviation
ds_std_deviation = dataset.std()

#Describe datset
dataset.describe()

#############################################Findidng Outliers##########################################
#Step 1. Calulate diff from mean
diff_from_mean = (dataset - ds_mean).abs()

#Step 2. Find all value which aren't within say 3 standard deviations from mean.
ds_outliers = dataset[diff_from_mean <= (3 * ds_std_deviation)] #It will replace all outliers with Nan

#Step 3. Drop outliers
ds_outliers.dropna(axis = 0)

