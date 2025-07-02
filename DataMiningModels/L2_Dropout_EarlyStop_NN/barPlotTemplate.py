import numpy as np
import matplotlib.pyplot as plt

def drawBarPlot(bars, group_labels, title="Bar Plot"):
    plt.figure(figsize=(8, 6))
    plt.bar(group_labels, bars, color='blue', edgecolor='white')
    plt.xlabel('Group')
    plt.ylabel('Value')
    plt.title(title)
    plt.tight_layout()

def drawGroupedBarPlot(bars1, bars2, bars3, group_labels, title="Grouped Bar Plot"):
    barWidth = 0.25
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.figure(figsize=(10, 6))
    plt.bar(r1, bars1, color='green', width=barWidth, edgecolor='white', label='var1')
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='var2')
    plt.bar(r3, bars3, color='black', width=barWidth, edgecolor='white', label='var3')

    plt.xlabel('Group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(group_labels))], group_labels)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
