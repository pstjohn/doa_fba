import pandas as pd
import numpy as np
import casadi as cs


## Growth Model

# Declare variables (use scalar graph)
t  = cs.SX.sym("t")    # time
# u  = SX.sym("u")    # control
x  = cs.SX.sym("x", 3)  # state
p = cs.SX.sym("p", 8) # Parameter

# ODE right hand side function

x_biomass = x[0]
x_glucose = x[1]

vmax = p[0]
km = p[1]

glucose_consumption = p[0] * x[1] / (p[1] + x[1]) * (1. / (1. + (x[2] * p[5])))

glucose_specific_growth = p[6]

xylose_consumption = p[2] * x[2] / (p[3] + x[2]) * (1. / (1. + (x[1] * p[4])))
xylose_specific_growth = p[7]

rhs = cs.vertcat([
    x_biomass * (glucose_specific_growth * glucose_consumption +
                 xylose_specific_growth * xylose_consumption),
    -x_biomass * glucose_consumption,
    -x_biomass * xylose_consumption,
])

f = cs.SXFunction('f', [t,x,p], [rhs])

model = cs.SXFunction('f', [t,x,p], [rhs])



from Collocation import Collocation

new = Collocation(model, ['biomass', 'glucose', 'xylose'])

opts = {
    'x0_max' : [0.1, 20, 50],
    'p_init' : 1.,
}

new.initialize(opts)

## Growth Data
ts = np.array([   0,   3,   6,  9,  12,  24, 36, 48, 59, 72, 96])
xs = np.array([[  8.7,  46.6],
                [  8.2,  45.2],
                [  6.2,  41.7],
                [  4. ,  35.4],
                [  1.8,  26.9],
                [  0.1,  13.1],
                [  0. ,   8.3],
                [  0. ,   4.3],
                [  np.nan ,   2.7],
                [  0. ,   1.4],
                [  0. ,   0.1]])
data = pd.DataFrame(xs, index=ts, columns=['glucose', 'xylose'])

new.set_data(data)
new.create_nlp()
out = new.solve(alpha=1E-3)



print out['f']
print out['p_opt']

import matplotlib.pylab as plt
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(out['ts'], out['x_opt'][:,1:], '.:')
ax[0].plot(out['ts'], out['x_sim'][:,1:], '-')
ax[0].plot(ts, xs, 'o')
ax[0].set_ylabel('Concentration')

ax[1].plot(out['ts'], out['x_opt'][:,0], '.:')
ax[1].plot(out['ts'], out['x_sim'][:,0], '-')

plt.show()

print out['x_opt'][0]
