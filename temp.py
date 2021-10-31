import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical


#import statistics
#import math
#
#a = np.array([1,2,3,4,5,6,7,8,9,10])
#
#mean = statistics.mean(a)
#total = 0
#for x in a :
#    total += (x - mean) ** 2 
#    
#stdev = math.sqrt(total / len(a) )    
#print(stdev)


#//////////////////
#data = np.loadtxt('test.csv', delimiter=',')
#mean=np.mean (data , axis=0)
#print(mean)


#RESHAPE test
#
#a = np.random.randint(0,10,20)
#print(a)
#
#b=np.reshape(a,(5,4))
#print(b)
 

#b = np.array([10,20,30,40,50])               
#c=b[b>30]
#print(c)


#fill values with zero which are less than the mean 
#a = np.array([10,20,30,40,50])
#a[a < np.mean(a)] = 0
#print(a)


#x = np.linspace( -10, 10, 100, dtype=np.float32)
#y1=np.sin(x)
#y2=np.cos(x)

#figure,axes = plt.subplots(nrows=3, ncols=3)
#figure.set_size_inches(10,10)

#axes[0,0].set_title('Sinüs Grafiği', fontweight='bold')
#axes[0,0].plot(x ,y1)

#axes[0,1].set_title('Kosinüs Grafiği', fontweight='bold')
#axes[0,1].plot(x ,y2)


# creating a dataframe 

# df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], columns=['X','Y','Z'] , index=['A','B','C'])
# print(df)


# creating a dataframe from a csv
# data = np.loadtxt('test.csv', dtype='float32', delimiter=',', usecols=[0,1,2], skiprows=1 )
# df = pd.DataFrame(data)
# print(df)


#GAUSS CURVE
#from scipy.stats import norm
#import numpy as np
#import matplotlib.pyplot as plt
#
#x = np.linspace(-5 ,5 , 1000)
#y = norm.pdf(x,0,1)
#plt.plot(x,y)


#### Artificial Neural Networks

#x = np.array([3, 5, 8, 9, 4])
#w = np.array([3, 6, 1, 7, 8])

#result = np.dot(x, w)
#print(result)

#result = np.sum(x * w)
#print(result)



#removing redundant columns from data tables
#dataset = np.loadtxt('test.csv', delimiter=',', skiprows=1, dtype='object')
#print(dataset)
#
#df = pd.read_csv('test.csv', usecols=[1,2])
#print(df)



#One Hot Encoding

#le = LabelEncoder()
#le.fit(['Kırmızı','Yeşil','Mavi'])
#data = le.transform(['Kırmızı','Yeşil','Mavi','Yeşil','Kırmızı'])
#print(data)




#df = pd.read_csv('test.csv')
#print(df)
#le = LabelEncoder()
#df['Renk'] = le.fit_transform(df['Renk'])
#print(df)
#dataset = df.to_numpy()
#print(dataset)



df = pd.read_csv('test.csv')
le = LabelEncoder()
df['Renk'] = le.fit_transform(df['Renk'])
df[['Renk-0','Renk-1','Renk2']] = to_categorical(df['Renk'])
df.drop(['Renk'], axis=1, inplace=True)
print(df)






































