import matplotlib.pyplot as plt
import numpy as np
import math
def example():
    plt.figure()
    theta = np.linspace(0,2*math.pi,100)
    r = 3*np.ones(100)
    plt.plot(theta,r,color='b')
    plt.show()

#plot 1:
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.subplot(1, 2, 1)
plt.plot(xpoints,ypoints)
plt.title("plot 1")

#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("plot 2")

plt.suptitle("RUNOOB subplot Test")
plt.show()