import numpy as np
import matplotlib.pyplot as plt


s1 = np.array([[1,0,0],[0,0,0],[0,-1,-1], [-1,-1,-1]])

s1_double = np.array([[[1,0,0],[0,0,0],[0,-1,-1], [-1,-1,-1]], [[0,0,0],[0,1,-1],[-1,-1,-1], [-1,1,-1]]])

print(s1.shape)
score = s1[~np.any(s1 == 1, axis = 1)]
print(score)
print(-np.sum(score))

print(np.all((np.array([0,0,0,0,0]) == 0)))
print(np.all((np.array([147456, 143886, 143886, 139937, 138049, 135318, 132780, 132215, 131809, 131728, 131639, 130974, 129101, 128166, 127030, 126860, 124795, 119486, 116528, 115380, 112888, 112326, 111234, 109176, 107274, 107068, 107212, 107210, 107210, 107210]) == 0)))

print((np.sum(s1 < 1)))


uncer = np.array([147456, 143886, 143886, 139937, 138049, 135318, 132780, 132215, 131809, 131728, 131639, 130974, 129101, 128166, 127030, 126860, 124795, 119486, 116528, 115380, 112888, 112326, 111234, 109176, 107274, 107068, 107212, 107210, 107210, 107210])


plt.plot(range(len(uncer)), uncer, 'r-')

plt.xlabel("Iterations")
plt.ylabel("Uncertainty")
plt.show()
# nmap = np.reshape(np.arange(9) - 10, (3, 3))


# nmap[2,1] = 8
# print(nmap)
# plt.imshow(nmap)
# ox, oy = np.where(nmap == 8)
# print(ox)
# print(oy)
# plt.plot(ox, oy, 'w.', markersize = 2) # pyplot and scatter indices is different y, x -> imshow (x, y)
# gx, gy = np.where(nmap == 5)
# plt.scatter(gx, gy, s = 10, color='r', marker='o') 


# plt.show()


