#ploting mean,std errors
import matplotlib.pyplot as plt
import pandas as pd

mean_errors_resnet = pd.read_csv('resnet_basic_60_epochs/test_mean.txt').values
mean_errors_zhang = pd.read_csv('zhang_basic_60_epochs/test_mean.txt').values

# Visualising the results
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Test mean error during epochs')
plt.title('Angle error of gaze predictions(in degrees)')
plt.xlabel('epochs')
plt.ylabel('angle error(degrees)')
plt.grid(alpha=0.5, linestyle=':')
plt.plot(mean_errors_resnet[:,1], mean_errors_resnet[:,2],color='blue',label='zhang')
plt.plot(mean_errors_zhang[:,1], mean_errors_zhang[:,2],color='red',label='resnet-3')
plt.ylim(bottom=4)
plt.legend()
plt.show()
