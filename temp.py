import numpy as np


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
a = np.array([10,20,30,40,50])
a[a < np.mean(a)] = 0
print(a)
