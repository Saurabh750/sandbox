import numpy as np

a = np.array([1,2,3])
# b = np.array([[4, 4],[5, 5],[6, 6]])
# c = np.array([4,5,6])
# print(np.dot(a, c))

# print(len(a))

output = np.array([0,1,1,0]).reshape(-1,1).T
print(output)
# trans = output.reshape(-1, 1)
# print(trans.T)