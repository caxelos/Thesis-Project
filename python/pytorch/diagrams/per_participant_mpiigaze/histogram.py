import matplotlib.pyplot as plt
import numpy as np
 
city=['p{:02}'.format(index) for index in range(15)]#['Delhi','Beijing','Washington','Tokyo','Moscow']
Gender=['Resnet-20 (ours)','GazeNet (best)']
pos = np.arange(len(city))
bar_width = 0.20#0.35
#resnet20=p00=[5.6,p01=7.2,p02=6.2,p03=7.2???,p04=7.4,p05=7.71,p06=7.0,p07=7.25,p08=7.3,p09=8.3,p10=6.46 ,p11=6.3,p12=6.57,p13=7.27,p14=6.55]
ok=[p00=[3.33,3.95,3.60,3.85,3.68],p01=[3.71,3.68,3.25,3.88,??],p02=[4.05,3.98,3.88,4.15,],p03=[??,??,??,5.0,4.66],p04=[4.12,4.09],p05=[],p06=[],p07=[],p08=[],p09=[],p10=[],p11=[],p12=[],p13=[],p14=[]]
gazenet=[]
 
plt.bar(pos,resnet20,bar_width,color='blue',edgecolor='black')
plt.bar(pos+bar_width,gazenet,bar_width,color='green',edgecolor='black')




plt.xticks(pos, city)
plt.xlabel('Participant', fontsize=14)
plt.ylabel('Mean error (in degrees)', fontsize=14)
plt.title('Mean error per participant. Train on UT Multiview. Test on MPIIGaze',fontsize=14)
plt.grid(alpha=0.5, linestyle=':')

plt.legend(Gender,loc=2)
plt.show()