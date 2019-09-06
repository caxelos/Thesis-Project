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
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Test mean error during epochs')
plt.title('Resnet-3: Angle error of gaze predictions(in degrees).')
plt.xlabel('epochs')
plt.ylabel('mean angle error(degrees)')
plt.grid(alpha=0.5, linestyle=':')

width=0.05
y,binEdges=np.histogram(mean_errors[:,1],bins=6)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
#plt.bar(bincenters,y, mean_errors[:,2], color='r', yerr=std_errors[:,2])

data       = mean_errors[:,1:3]#np.array(np.random.rand(1000))
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
menStd     = np.sqrt(y)
width      = 4

plt.bar(mean_errors[:,1], mean_errors[:,2],width, yerr=std_errors[:,2])
plt.legend('mean error','standar deviation')
plt.show()

#menStd     = np.sqrt(y)
#width      = 0.05
#plt.bar(bincenters, y, width=width, color='r', yerr=menStd)
