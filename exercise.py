import matplotlib.pyplot as plt
import numpy as np
import math
jingdu = 100
r = 4*np.ones(jingdu)
theta = np.linspace(0,2*math.pi,jingdu)
ax = plt.subplot(111,projection='polar')
ax.plot(theta,r)
ax.plot(theta,2*r)
ax.plot(theta,3*r)
ax.grid(False)
plt.show()