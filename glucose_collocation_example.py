import pandas as pd
import numpy as np
import casadi as cs


## Growth Model

# Declare variables
t = cs.SX.sym("t")    # time
x = cs.SX.sym("x", 2) # state
p = cs.SX.sym("p", 2) # parameters

# Declare dynamic system
x_biomass = x[0]

vmax = p[0]
km = p[1]

glucose_specific_growth = 0.040678


# MM uptake kinetics.
glucose_consumption = p[0] * x[1] / (p[1] + x[1])

rhs = cs.vertcat([
    x_biomass * (glucose_specific_growth * glucose_consumption),
    -x_biomass * glucose_consumption,
])

model = cs.SXFunction('f', [t,x,p],[rhs])



from Collocation import Collocation

new = Collocation(model, ['biomass', 'glucose'])

opts = {
    'x0_max' : [0.1, 20],
    'p_init' : 1.,
}

new.initialize(opts)

## Growth Data
ts = np.array([   0,   3,   6,  9,  12,  24, 36, 48, 59, 72, 96])
xs = np.array([ 8.7, 8.2, 6.2, 4., 1.8, 0.1, 0., 0., 0., 0., 0.])
data = pd.DataFrame(xs, index=ts, columns=['glucose'])

new.set_data(data)
new.create_nlp()
out = new.solve(alpha=1E-3)



print out['f']
print out['p_opt']

import matplotlib.pylab as plt
plt.plot(data.index, data, 'o')
plt.plot(out['ts'], out['x_opt'][:,1:], '.:')
plt.plot(out['ts'], out['x_sim'][:,1:], '-')
plt.show()
