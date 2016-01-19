import pandas as pd
import numpy as np

import casadi as cs

from .VariableHandler import VariableHandler
from .BaseCollocation import BaseCollocation

from warnings import warn


class Collocation(BaseCollocation):

    def __init__(self, model, boundary_species):
        """ Initialize the collocation object.

        model: a cs.SXFunction object
            a model that describes substrate uptake and biomass formation
            kinetics. Inputs should be [t, x, p], outputs should be [ode]

        boundary_species: list
            List of string ID's for each of the states in the model. The first
            state should represent the current biomass concentration.

        """

        # Assign sizing variables
        self.nx = model.getInput(1).shape[0]
        self.np = model.getInput(2).shape[0]

        assert model.getOutput(0).shape[0] == self.nx, \
            "Output length mismatch"

        # Attach model
        self.dxdt = model

        # Attach state names
        assert len(boundary_species) == self.nx, "Name length mismatch"
        self.boundary_species = np.asarray(boundary_species)

        super(Collocation, self).__init__()

        # setup defaults
        self.tf = 100.
        
    def setup(self):
        """ Set up the collocation framework """

        self._initialize_polynomial_coefs()
        self._initialize_variables()
        self._initialize_polynomial_constraints()
        
    def initialize(self, **kwargs):
        """ Call after setting up boundary kinetics, finalizes the
        initialization and sets up the NLP problem. Keyword arguments are
        passed directly as options to the NLP solver """

        self._initialize_mav_objective()
        self._initialize_solver(**kwargs)

    def _initialize_variables(self):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'p'  : (self.np),
        }

        self.var = VariableHandler(core_variables)
        
        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 200.

        self.var.p_lb[:] = 0.
        self.var.p_ub[:] = 100.
        
        
    def _initialize_polynomial_constraints(self):
        """ Add constraints to the model to account for system dynamics and
        continuity constraints """

        h = self.tf / self.nk

        # All collocation time points
        T = np.zeros((self.nk, self.d+1), dtype=object)
        for k in range(self.nk):
            for j in range(self.d+1):
                T[k,j] = h*(k + self.col_vars['tau_root'][j])


        # For all finite elements
        for k in range(self.nk):

            # For all collocation points
            for j in range(1, self.d+1):

                # Get an expression for the state derivative at the collocation
                # point
                xp_jk = 0
                for r in range(self.d+1):
                    xp_jk += self.col_vars['C'][r,j]*cs.SX(self.var.x_sx[k,r])

                # Add collocation equations to the NLP.
                # (Pull boundary fluxes for this FE from the flux DF)
                [fk] = self.dxdt.call(
                    [T[k,j], cs.SX(self.var.x_sx[k,j]), cs.SX(self.var.p_sx)])

                self.constraints_sx.append(h*fk - xp_jk)
                self.constraints_lb.append(np.zeros(self.nx))
                self.constraints_ub.append(np.zeros(self.nx))

            # Add continuity equation to NLP
            if k+1 != self.nk:
                
                # Get an expression for the state at the end of the finite
                # element
                xf_k = self.col_vars['D'].dot(cs.SX(self.var.x_sx[k]))

                self.constraints_sx.append(cs.SX(self.var.x_sx[k+1,0]) - xf_k)
                self.constraints_lb.append(np.zeros(self.nx))
                self.constraints_ub.append(np.zeros(self.nx))

        # Get an expression for the endpoint for objective purposes
        xf = self.col_vars['D'].dot(cs.SX(self.var.x_sx[-1]))
        self.xf = {met : x_sx for met, x_sx in zip(self.boundary_species, xf)}

    def _initialize_mav_objective(self):
        """ Initialize the objective function to minimize the absolute value of
        the flux vector """

        self.objective_sx += (self.col_vars['alpha'] *
                              cs.fabs(self.var.p_sx).sum())


    def _plot_setup(self):

        # Create vectors from optimized time and states
        h = self.tf / self.nk

        self.fs = h * np.arange(self.nk)
        self.ts = np.array(
            [point + h*np.array(self.col_vars['tau_root']) for point in 
             np.linspace(0, self.tf, self.nk,
                         endpoint=False)]).flatten()

        self.sol = self.var.x_op.reshape((self.nk*(self.d+1)), self.nx)


    def _get_interp(self, t, states=None):
        """ Return a symbolic polynomial representation of the state vector
        evaluated at time t.

        states: list
            indicies of which states to return

        """

        assert t < self.tf, "Final time must be increased"

        h = self.tf / self.nk

        if not states: states = xrange(1, self.nx)

        finite_element = int(t / h)
        tau = (t % h) / h
        basis = self.col_vars['lfcn']([tau])[0].toArray().flatten()
        x_roots = self.var.x_sx[finite_element, :, states]

        return np.inner(basis, x_roots)

    def set_data(self, data, weights=None):
        """ Attach experimental measurement data.

        data : a pd.DataFrame object
            Data should have columns corresponding to the state labels in
            self.boundary_species, with an index corresponding to the measurement
            times.

        """

        # Should raise an error if no state name is present
        df = data.loc[:, self.boundary_species]

        # Rename columns with state indicies
        df.columns = np.arange(self.nx)

        # Remove empty (nonmeasured) states
        self.data = df.loc[:, ~pd.isnull(df).all(0)]

        if weights is None:
            weights = self.data.max()

        obj_list = []
        for ((ti, state), xi) in self.data.stack().iteritems():
            obj_list += [(self._get_interp(ti, [state]) - xi) / weights[state]]

        obj_resid = cs.sum_square(cs.vertcat(obj_list))
        self.objective_sx += obj_resid


    def solve(self, ode=True, **kwargs):

        out = super(Collocation, self).solve(**kwargs)
        if ode: self.solve_ode()

        return out


    def solve_ode(self):
        """ Solve the ODE using casadi's CVODES wrapper to ensure that the
        collocated dynamics match the error-controlled dynamics of the ODE """


        self.ts.sort() # Assert ts is increasing

        f_integrator = cs.SXFunction('ode',
                                     cs.daeIn(
                                         t = self.dxdt.inputExpr(0),
                                         x = self.dxdt.inputExpr(1),
                                         p = self.dxdt.inputExpr(2)),
                                     cs.daeOut(
                                         ode = self.dxdt.outputExpr(0)))

        integrator = cs.Integrator('int', 'cvodes', f_integrator)
        simulator = cs.Simulator('sim', integrator, self.ts)
        simulator.setInput(self.sol[0], 'x0')
        simulator.setInput(self.var.p_op, 'p')
        simulator.evaluate()
        x_sim = self.sol_sim = np.array(simulator.getOutput()).T

        err = ((self.sol - x_sim).mean(0) /
               (self.sol.mean(0))).mean()

        if err > 1E-3: warn(
                'Collocation does not match ODE Solution: \
                {:.2f}% Error'.format(100*err))


    def plot_optimal(self):

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set_context('talk', font_scale=1.5)
        sns.set(color_codes=True)

        fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1,
                               figsize=(8,5))
        
        lines = ax[0].plot(self.ts, self.sol[:,1:], '.--')
        ax[0].legend(lines, self.boundary_species[1:],
                     loc='upper center', ncol=2)
        state_data = self.data.loc[:, self.data.columns > 0]
        ax[0].plot(state_data.index, state_data, 'o')

        lines = ax[1].plot(self.ts, self.sol[:,0], '.--')
        ax[1].legend(lines, self.boundary_species[0], loc='upper left',
                     ncol=1)
        bio_data = self.data.loc[:, self.data.columns == 0]
        if not bio_data.empty:
            ax[1].plot(bio_data.index, bio_data, 'o')
        
        plt.show()
        
