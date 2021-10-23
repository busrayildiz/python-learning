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




data = np.loadtxt('test.csv', delimiter=',')
mean=np.mean (data , axis=0)
print(mean)