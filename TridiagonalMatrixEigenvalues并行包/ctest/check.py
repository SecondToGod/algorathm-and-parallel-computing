import numpy as np 

f = np.loadtxt('./data67.txt')
x = np.matrix(f)
print x
a = np.linalg.matrix_rank(x)
print ('matrix rank:' , a)
b = []
c =[[]]
b ,c = np.linalg.eig(x)
b.sort()
print ('eigenvalues are:\n', b)
print ('eigenVectors are:\n',c)



