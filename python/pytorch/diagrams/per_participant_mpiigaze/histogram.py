import matplotlib.pyplot as plt
import numpy as np
 
city=['p{:02}'.format(index) for index in range(15)]#['Delhi','Beijing','Washington','Tokyo','Moscow']
Gender=['Resnet-20 (ours)','GazeNet (best)']
pos = np.arange(len(city))
bar_width = 0.20#0.35
#resnet20=p00=[5.6,p01=7.2,p02=6.2,p03=7.2???,p04=7.4,p05=7.71,p06=7.0,p07=7.25,p08=7.3,p09=8.3,p10=6.46 ,p11=6.3,p12=6.57,p13=7.27,p14=6.55]
#ok=[p00=[3.33,3.95,3.60,3.85,3.68],p01=[3.71,3.68,3.25,3.88,4.08],p02=[4.05,3.98,3.88,4.15,3.81],p03=[4.55,3.19,3.88,5.0,4.66],p04[[4.10,4.12,4.09,4.57,4.49],p05=[4.65,5.2,4.75,4.55,4.22 ],p06=[4.95,5.81,5.15,4.65,4.5],p07=[6.18,6.36,6.53,??,6.188],p08=[6.12,??,6.57],p09=[],p10=[4.04,6.35,6.51,6.54,6.25],p11=[??,4.37,4.23,],p12=[],p13=[7.3,5.7,6.84,6.85,8.15],p14=[5.0,4.15,5.41,4.82,7.20]]
gazenet=[1.7,2,2.1,2,2.2,2.9,3.1,4,4,2.8,2,2.5,3,2.7,2]
gazenetstd=[1.3,2,1.7,2,1.7,1.8,2,3,3,1.8,1.8,1.7,2,1.9,1.7]
 
plt.bar(pos,gazenet,bar_width,yerr=gazenetstd,alpha=0.5,color='blue',edgecolor='black',capsize=3)
plt.bar(pos+bar_width,gazenet,bar_width,color='green',edgecolor='black')




plt.xticks(pos, city)
plt.xlabel('Participant', fontsize=14)
plt.ylabel('Mean error (in degrees)', fontsize=14)
plt.title('Mean error per participant. Train on UT Multiview. Test on MPIIGaze',fontsize=14)
plt.grid(alpha=0.5, linestyle=':')

plt.legend(Gender,loc=2)
plt.show()