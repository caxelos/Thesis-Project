import matplotlib.pyplot as plt
import numpy as np
 
city=['p{:02}'.format(index) for index in range(1,16)]#['Delhi','Beijing','Washington','Tokyo','Moscow']
Gender=['Resnet-20 (ours)','GazeNet (best)']
pos = np.arange(len(city))
bar_width = 0.22#0.35
#resnet20=p00=[5.6,p01=7.2,p02=6.2,p03=7.2???,p04=7.4,p05=7.71,p06=7.0,p07=7.25,p08=7.3,p09=8.3,p10=6.46 ,p11=6.3,p12=6.57,p13=7.27,p14=6.55]
#resnet20=np.array([[3.33,3.95,3.60,3.85,3.68],[3.71,3.68,3.25,3.88,4.08],[4.05,3.98,3.88,4.15,3.81],[4.55,3.19,3.88,5.0,4.66],[[4.10,4.12,4.09,4.57,4.49],[4.65,5.2,4.75,4.55,4.22 ],[4.95,5.81,5.15,4.65,4.5],[6.18,6.36,6.53,6.59,6.188],[6.12,6.4,6.57,6.03,6.48],[5.6,4.9,4.76,5.47,5.75],[4.04,6.35,6.51,6.54,6.25],p11=[4.57,4.37,4.23,6.32,4.34],[5.64,6.05,7.39,5.73,6.3],[7.3,5.7,6.84,6.85,8.15],[5.0,4.15,5.41,4.82,7.20]]
gazenet=np.array([1.7,2,2.1,2,2.2,2.9,3.1,4,4,2.8,2,2.5,3,2.7,2])
gazenetstd=np.array([1.3,1.8,1.7,1.9,1.7,1.9,2.0,3.0,2.8,1.8,1.8,1.7,2,1.9,1.7])
 
plt.bar(pos,gazenet,bar_width,yerr=gazenetstd,alpha=0.5,color='blue',edgecolor='black',capsize=3)
#plt.bar(pos+bar_width,gazenet,bar_width,color='green',edgecolor='black')




plt.xticks(pos, city)
plt.xlabel('Participant', fontsize=14)
plt.ylabel('Mean error (in degrees)', fontsize=14)

# per participant
#plt.title('Mean error per participant. Train on UT Multiview. Test on MPIIGaze',fontsize=14)
# person specific
plt.title('Mean error (person specific). Train and Test on MPIIGaze',fontsize=14)


plt.grid(alpha=0.5, linestyle=':')

plt.legend(Gender,loc=2)
plt.show()