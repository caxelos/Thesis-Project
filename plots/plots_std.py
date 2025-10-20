#ploting mean,std errors
import matplotlib.pyplot as plt
import pandas as pd

resnet8_preact_basicBlock = pd.read_csv('resnet8_preact_basic/test_stdev.txt').values
zhang = pd.read_csv('zhang_basic_60_epochs/test_stdev.txt').values
resnet6_basic = pd.read_csv('resnet6_basic/test_stdev.txt').values
resnet8_preact_bottleneckBlock = pd.read_csv('resnet8_preact_bottleneck/test_stdev.txt').values
# Visualising the results
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Test mean error during epochs')
plt.title('standard deviation angle error of gaze predictions(in degrees)')
plt.xlabel('epochs')
plt.ylabel('angle error(degrees)')
plt.grid(alpha=0.5, linestyle='-',linewidth=1)

plt.plot(resnet8_preact_basicBlock[:,1],resnet8_preact_basicBlock[:,2],color='cyan',label='resnet-8 preact(basic block)')
plt.plot(zhang[:,1], zhang[:,2],color='red',label='zhang')
plt.plot(resnet6_basic[:,1], resnet6_basic[:,2],color='blue',label='resnet-6 basic')
plt.plot(resnet8_preact_bottleneckBlock[:,1], resnet8_preact_bottleneckBlock[:,2],color='green',label='resnet-8 preact(bottleneck)')


plt.ylim(bottom=2)
plt.legend()
plt.show()
