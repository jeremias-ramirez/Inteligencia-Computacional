import numpy as np
import backpropagation as bp

w = []
w.append( 0.5 * np.ones((2,3)))
w.append( 0.5 * np.ones((1,3)))

y = []
y.append( np.array([[-1],[1],[1]]) )
y.append( np.array([[-1],[0.5],[0.5]]) )
y.append( np.array([[-1],[0.5]]) )
yd = -np.ones((1,1))
vel = 1

wNew = bp.backpropagation(w,y,yd,vel)
print(wNew)

