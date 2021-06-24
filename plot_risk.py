import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(15,4))
y = np.loadtxt('./inference/output/results_risk.txt')
y2 = np.loadtxt('./inference/output/results_risk2.txt')
x = np.arange(0,200)
plt.ylim(0,1)
plt.xlabel('frame')
plt.ylabel('score')
plt.plot(x,y[0:200],'*-',color ='r',label='prediction')
plt.plot(x,y2[0:200],'-',color = 'b',label = 'ground truth')
plt.savefig('risk_plot.png')
