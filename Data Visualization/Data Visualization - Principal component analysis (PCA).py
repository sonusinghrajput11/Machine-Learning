import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.decomposition import PCA

#Reading iris dataset
iris_dataset = load_iris()

#Reading features
features = iris_dataset.data

#Reading targets
target = iris_dataset.target

#Reading targets name
target_name = iris_dataset.target_names

#Reading features name
features_name = iris_dataset.feature_names

#Creating Data Frames
dataset = pd.DataFrame(data = features, columns = features_name)

#Association target value for each sample
dataset['labelDesc'] = target_name[target]
dataset['label'] = target

#Suffle the dataset
dataset_random = dataset.sample(frac=1)

#scatter plot using pandas

ax = dataset_random[dataset_random.labelDesc =='setosa'].plot.scatter(x='sepal length (cm)', y='sepal width (cm)', 
                                                    color='red', label='setosa')
dataset_random[dataset_random.labelDesc=='versicolor'].plot.scatter(x='sepal length (cm)', y='sepal width (cm)', 
                                                color='green', label='versicolor', ax=ax)
dataset_random[dataset_random.labelDesc=='virginica'].plot.scatter(x='sepal length (cm)', y='sepal width (cm)', 
                                                color='blue', label='virginica', ax=ax)
ax.set_title("scatter")

#Paired plot using seaborn
sns.set()
sns.pairplot(dataset_random[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'labelDesc']],
             hue="labelDesc", diag_kind="kde")

#PCA analysis using scikit learn
pca = PCA(n_components=2)
dataset_random['pca1'] = pca.fit(features, target).transform(features)[:, 0]
dataset_random['pca2'] = pca.fit(features, target).transform(features)[:, 1]


#scatter plot using pandas after PCA

ax = dataset_random[dataset_random.labelDesc =='setosa'].plot.scatter(x='pca1', y='pca2', 
                                                    color='red', label='setosa')
dataset_random[dataset_random.labelDesc=='versicolor'].plot.scatter(x='pca1', y='pca2', 
                                                color='green', label='versicolor', ax=ax)
dataset_random[dataset_random.labelDesc=='virginica'].plot.scatter(x='pca1', y='pca2', 
                                                color='blue', label='virginica', ax=ax)
ax.set_title("scatter")

#Scatter plot using matplot lib
plt.scatter(dataset_random.pca1, dataset_random.pca2, c = dataset_random.label)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

