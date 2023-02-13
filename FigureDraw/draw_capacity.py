import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

x = [2, 3, 4, 5]
y = [25, 31, 39, 39]

fig, ax = plt.subplots()

# Be sure to only pick integer tick locations.
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))

# Plot anything (note the non-integer min-max values)...
ax.plot(x, y, color='black')
ax.scatter(x, y, color='black')

ax.text(x=2.05, y=24.8, s="(2, 25)", fontname = "Arial", fontsize=14)
ax.text(x=3.05, y=30.8, s="(3, 31)", fontname = "Arial", fontsize=14)
ax.text(x=3.6, y=38.7, s="(4, 39)", fontname = "Arial", fontsize=14)
ax.text(x=4.8, y=38.2, s="(5, 39)", fontname = "Arial", fontsize=14)

plt.xticks(fontsize=14, fontname = "Arial")
plt.yticks(fontsize=14, fontname = "Arial")
plt.xlabel("Number of larvae in each well", fontname = "Arial", fontsize=14)
plt.ylabel("Number of valid videos", fontname = "Arial", fontsize=14)
# Just for appearance's sake
ax.margins(0.05)
ax.axis('tight')
fig.tight_layout()
plt.savefig("./figure_results/verification_capacity.eps", format='eps')