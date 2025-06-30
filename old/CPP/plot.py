#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

data1 = np.loadtxt('u.txt')

m = int(data1[0])   # number of nodes
t = int(data1[1])   # number of time steps

data2 = data1[2:]

u_sol = data2.reshape(t, m)
x = np.linspace(0, 2, m)

for i in xrange(0, t):
    plt.plot(x, u_sol[i], '-o')

axes = plt.gca()
axes.set_ylim([-0.1,1.1])

plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$u(x)$', fontsize = 14)
plt.savefig('u.pdf')
plt.show()
