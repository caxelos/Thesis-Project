#ploting mean,std errors
#import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
import numpy as np
skips=[]
for i in range(60):
    if i%10 != 0:
        skips.append(i)
mean_errors = pd.read_csv('test_mean.txt',skiprows=skips).values
std_errors=pd.read_csv('test_stdev.txt',skiprows=skips).values
#print(mean_errors[:,1:])

#data = dataset_train.iloc[:, 1:2].values

# Visualising the results
plt.title('Resnet-8 preact using bottleneck block: Angle error of gaze predictions(in degrees).')
plt.xlabel('epochs')
plt.ylabel('mean angle error(degrees)')
plt.grid(alpha=0.5, linestyle=':')

width=0.05
y,binEdges=np.histogram(mean_errors[:,1],bins=6)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
data       = mean_errors[:,1:3]#np.array(np.random.rand(1000))
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
menStd     = np.sqrt(y)
width      = 4

plt.bar(mean_errors[:,1], mean_errors[:,2],width, yerr=std_errors[:,2],alpha=0.5, ecolor='black', capsize=3)
plt.legend('mean error','standar deviation')
plt.show()

# Save the figure and show
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
#plt.show()