import numpy as np
import matplotlib.pyplot as plt

train_pts = 30

# 创建测试的数据点，2类  
# 以(-1.5, -1.5)为中心  
rand1=np.ones((train_pts,2))*(-2)+np.random.rand(train_pts,2)
print('rand1：')
print(rand1)

# 以(1.5, 1.5)为中心  
rand2=np.ones((train_pts,2))+np.random.rand(train_pts,2)
print('rand2:')
print(rand2)

# 合并随机点，得到训练数据  
train_data=np.vstack((rand1,rand2))
train_data=np.array(train_data,dtype='float32')
train_label=np.vstack((np.zeros((train_pts,1),dtype='int32'),np.ones((train_pts,1),dtype='int32')))
# 显示训练数据  
plt.figure(1)
plt.plot(rand1[:,0],rand1[:,1],'o')
plt.plot(rand2[:,0],rand2[:,1],'o')
plt.plot(rand2[:,0],rand2[:,1],'o')


