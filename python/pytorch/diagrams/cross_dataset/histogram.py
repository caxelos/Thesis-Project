import matplotlib.pyplot as plt
import numpy as np
 
city=['p{:02}'.format(index) for index in range(15)]#['Delhi','Beijing','Washington','Tokyo','Moscow']
Gender=['Resnet-20 (ours)','GazeNet (best)']
pos = np.arange(len(city))
bar_width = 0.20#0.35
resnet20=[8.89,11.64,10.3,17.24,12.03,12.15,11.29,13.11,12.85,11.96,10.28,13.27,17.32,12.86,11.47]
gazenet=[6.5,7.7,7.6,8.0,9.6,7,8.4,9.7,7.2,9.2,8.5,5.8,7.9,6.9,8.8]
 
plt.bar(pos,resnet20,bar_width,color='blue',edgecolor='black')
plt.bar(pos+bar_width,gazenet,bar_width,color='green',edgecolor='black')
plt.xticks(pos, city)
plt.xlabel('Participant', fontsize=14)
plt.ylabel('Mean error (in degrees)', fontsize=14)
plt.title('Mean error per participant. Train on UT Multiview. Test on MPIIGaze',fontsize=14)
plt.grid(alpha=0.5, linestyle=':')

plt.legend(Gender,loc=2)
plt.show()