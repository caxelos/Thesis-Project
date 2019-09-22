#ploting mean,std errors
import matplotlib.pyplot as plt
import pandas as pd

#std_errors_resnet = pd.read_csv('resnet_basic_60_epochs/test_stdev.txt').values
std_errors_zhang = pd.read_csv('zhang_basic_60_epochs/test_stdev.txt').values
std_errors_resnet6_basic = pd.read_csv('resnet6_basic/test_stdev.txt').values
std_errors_resnet8_preact = pd.read_csv('resnet8_preact/test_stdev.txt').values
# Visualising the results
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Test mean error during epochs')
plt.title('standard deviation angle error of gaze predictions(in degrees)')
plt.xlabel('epochs')
plt.ylabel('angle error(degrees)')
plt.grid(alpha=0.5, linestyle='-',linewidth=1)
#plt.plot(std_errors_resnet[:,1], std_errors_resnet[:,2],color='blue',label='resnet-3')
plt.plot(std_errors_zhang[:,1], std_errors_zhang[:,2],color='red',label='zhang')
plt.plot(std_errors_resnet6_basic[:,1], std_errors_resnet6_basic[:,2],color='blue',label='resnet-6 basic')
plt.plot(std_errors_resnet8_preact[:,1], std_errors_resnet8_preact[:,2],color='green',label='resnet-8 preact')


plt.ylim(bottom=2)
plt.legend()
plt.show()
