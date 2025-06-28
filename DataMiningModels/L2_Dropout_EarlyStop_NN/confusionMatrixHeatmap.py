import matplotlib.pyplot  as plt
import numpy as np
import matplotlib

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
ideal_matrix = np.array([[0.9,0.1,0.1], [0.1,0.9,0], [0,0,1]])
sns.set()

print (matplotlib.__version__)

ax = sns.heatmap(ideal_matrix, annot=True, fmt='.1f',linewidth=0.5)
plt.xlabel('True Class')
plt.ylabel('Predicted Class')

#plt.savefig("hm.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf
plt.show()