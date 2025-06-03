# Machine Learning
# Wine Analysis
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics

raw_data = datasets.load_wine()
print('data.shape\t',raw_data['data'].shape,
         '\ntarget.shape \t',raw_data['target'].shape)
features = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
print(features.head())
data = features
data['target'] = raw_data['target']
data['class'] = data['target'].map(lambda ind: raw_data['target_names'][ind])
print(data.head())

alcoholDist = sns.histplot(data['alcohol'], kde=False)
alcoholFig = alcoholDist.get_figure()
alcoholFig.savefig("alcoholDist.pdf")
plt.clf()

for i in data.target.unique():
   sns.histplot(data['alcohol'][data.target==i], kde=True, label='{}'.format(i))
plt.legend()
plt.savefig('classDist.pdf')

import matplotlib.gridspec as gridspec

for feature in raw_data['feature_names']:
    print(feature)
    gs1 = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs1[:-1])
    ax2 = plt.subplot(gs1[-1])
    gs1.update(right=0.60)
    sns.boxplot(x=feature, y='class', data=data, ax=ax2)
    sns.kdeplot(data[feature][data.target==0], ax=ax1, label='0')
    sns.kdeplot(data[feature][data.target==1], ax=ax1, label='1')
    sns.kdeplot(data[feature][data.target==2], ax=ax1, label='2')
    ax2.yaxis.label.set_visible(False)
    ax1.xaxis.set_visible(False)
    plt.show()

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = \
       train_test_split(raw_data['data'], raw_data['target'], test_size=0.2)

print(len(data_train), ' samples in training data\n',
         len(data_test), ' samples in test data\n', )

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf = clf.fit(data_train, label_train)

plt.clf()
p = tree.plot_tree(clf.fit(data_train, label_train))
plt.show()

from sklearn import metrics

y_pred = clf.predict(data_test)
print(metrics.accuracy_score(label_test, y_pred))