import numpy as np
from glob import glob
import pickle

cs = ['F' , 'N' , 'O' , 'S' , 'Z']

x = []
y = []

for ind in range(len(cs)):
    data_dir = './data/' + cs[ind] + '/*.txt'
    all_f = glob(data_dir)
    if(len(all_f) == 0):
        data_dir = './data/' + cs[ind] + '/*.TXT'
        all_f = glob(data_dir)
    # print(all_f)
    for f in all_f:
        a = open(f,'r')
        data = [float(i) for i in a.readlines()]
        x.append(data)
        y.append(ind)
        # print(data)
        # print(len(data))

x = np.array(x)
y = np.array(y)
print(x.shape)

pickle.dump(x, open('x.pkl' , 'wb'))
pickle.dump(y, open('y.pkl' , 'wb'))