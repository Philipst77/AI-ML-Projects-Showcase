# Print a bar chart with groups

import numpy as np
import matplotlib.pyplot as plt

# set height of bar
# length of these lists determine the number
# of groups (they must all be the same length)
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]

# set width of bar. To work and supply some padding
# the number of groups times barWidth must be
# a little less than 1 (since the next group
# will start at 1, then 2, etc).

barWidth = 0.25
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='black', width=barWidth, edgecolor='white', label='var3')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

# Create legend & Show graphic
plt.legend()
plt.show()
#plt.savefig("barChart.pdf",dpi=400,bbox_inches='tight',pad_inches=0.05) # save as a pdf