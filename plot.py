import matplotlib.pyplot as plt
import re
import math
import numpy as np

####plot loss curve
f = open('nohup.out','r')
txt = f.read()
loss_list_train = re.findall(r'Training loss: \d+\.?\d+',txt)
loss_list_test = re.findall(r'Testing loss: \d+\.?\d+',txt)
#print(loss_list_train)
#print(loss_list_test)
loss_list_train1 = [float(s.strip('Training loss: ')) for s in loss_list_train]
loss_list_test1 = [float(s.strip('Testing loss: ')) for s in loss_list_test]

#loss_int = list(map(float,loss_list_train1))
#loss_int1 = []
#for n in range(18,len(loss_int),19):
#    loss_int1.append(loss_int[n])

#print(loss_int)
plt.figure()
loss_list_train1 = [x for x in loss_list_train1]
loss_list_test1 = [x for x in loss_list_test1]
plt.plot(range(1,len(loss_list_train1)+1),loss_list_train1, '-o')
plt.plot(range(1,len(loss_list_test1)+1),loss_list_test1, '-o')
#plt.title('Loss Curve', fontsize=22) 
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Loss', fontsize = 18)
plt.tick_params(axis= 'x', direction='in', labelsize = 16)
plt.tick_params(axis= 'y', direction='in', labelsize = 16)
plt.xticks(np.arange(0,len(loss_list_train1)+1,step=2))
plt.show()


