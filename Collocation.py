from warnings import warn

import numpy as np
from scipy.interpolate import lagrange
import pandas as pd

import casadi as cs

from .VariableHandler import VariableHandler
from .BaseCollocation import BaseCollocation

class Collocation(BaseCollocation):

    def __init__(self, model, boundary_species):
        """ A class to handle the optimization of kinetic uptake parameters to
        match a dynamic model to a given set of experimental data.

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
        self.var.x_in[:] = 1.

        self.var.p_lb[:] = 0.
        self.var.p_ub[:] = 100.
        self.var.p_in[:] = 0.

        # Initialize optimization parameters
        parameters = {
            'alpha' : (1,),
        }

        self.pvar = VariableHandler(parameters)
        self.pvar.alpha_in[:] = 0.
        
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
                # Boundary fluxes are calculated by multiplying the EFM
                # coefficients in V by the efm matrix
                [fk] = self.dxdt.call(
                    [T[k,j], cs.SX(self.var.x_sx[k,j]), cs.SX(self.var.p_sx)])

                self.add_constraint(h * fk - xp_jk)

            # Add continuity equation to NLP
            if k+1 != self.nk:
                
                # Get an expression for the state at the end of the finite
                # element
                xf_k = self.col_vars['D'].dot(cs.SX(self.var.x_sx[k]))
                self.add_constraint(cs.SX(self.var.x_sx[k+1,0]) - xf_k)

        # Get an expression for the endpoint for objective purposes
        xf = self.col_vars['D'].dot(cs.SX(self.var.x_sx[-1]))
        self.xf = {met : x_sx for met, x_sx in zip(self.boundary_species, xf)}

        # Similarly, get an expression for the beginning point
        x0 = self.var.x_sx[0,0,:]
        self.x0 = {met : x_sx for met, x_sx in zip(self.boundary_species, x0)}    

    def _initialize_mav_objective(self):
        """ Initialize the objective function to minimize the absolute value of
        the flux vector """

        self.objective_sx += (self.pvar.alpha_sx *
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


    def _get_interp(self, t, states=None, x_rep='sx'):
        """ Return a polynomial representation of the state vector
        evaluated at time t.

        states: list
            indicies of which states to return

        x_rep: 'sx' or 'op', most likely.
            whether or not to interpolate symbolic or optimal values of the
            state variable

        """

        assert t < self.tf, "Requested time is outside of the simulation range"

        h = self.tf / self.nk

        if states is None: states = xrange(1, self.nx)

        finite_element = int(t / h)
        tau = (t % h) / h
        basis = self.col_vars['lfcn']([tau])[0].toArray().flatten()
        x = getattr(self.var, 'x_' + x_rep)
        x_roots = x[finite_element, :, states]

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

        self.weights = weights

        self._set_objective_from_data(self.data, self.weights)


    def _set_objective_from_data(self, data, weights):

        obj_list = []
        for ((ti, state), xi) in data.stack().iteritems():
            obj_list += [(self._get_interp(ti, [state]) - xi) / weights[state]]

        obj_resid = cs.sum_square(cs.vertcat(obj_list))
        self.objective_sx += obj_resid


    def solve(self, ode=True, **kwargs):

        out = super(Collocation, self).solve(**kwargs)
        if ode: self.solve_ode()

        return out

    def warm_solve(self, ode=True, **kwargs):

        out = super(Collocation, self).warm_solve(**kwargs)
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

        colors = sns.color_palette()
        
        ts = np.linspace(0, self.tf, 100)
        sol = self._interpolate_solution(ts)

        lines = ax[0].plot(ts, sol[:,1:], '-')
        ax[0].legend(lines, self.boundary_species[1:],
                     loc='upper center', ncol=2)

        lines = ax[1].plot(ts, sol[:,0], '-', color=colors[0])
        ax[1].legend(['Biomass'], loc='upper center')


        state_data = self.data.loc[:, self.data.columns > 0]
        bio_data = self.data.loc[:, self.data.columns == 0]

        for (name, col), color in zip(state_data.iteritems(), colors):
            ax[0].plot(col.index, col, 'o', color=color)

        if not bio_data.empty:
            ax[1].plot(bio_data.index, bio_data, 'o', color=colors[0])
        
        plt.show()
        
        return fig

    def reset_objective(self):
        self.objective_sx = 0


    def bootstrap(self, n, sample_size=None, map_fn=None, solve_method='warm',
                  progress=True, **solve_opts):
        """Perform a bootstrap analysis given the current data. Resamples (with
        replacement) `n` number of times and re-estimates parameters.

        n (int): number of bootstrap samples
        sample_size (int): size of each bootstrap sample.
            Defaults to current data size
        map_fn (function): A function to apply to the each bootstrap sample.
            should accept a collocation class as input. Defaults to simply
            returning self.var.vars_op
        solve_method ('warm' or None): whether or not to use warm solves
        progess (bool): whether or not to show a progress bar
        
        """
        if sample_size == None: sample_size = self.data.shape[0] 
        if map_fn == None: map_fn = lambda coll: coll.var.vars_op

        results = []

        if progress:
            import progressbar
            pbar = progressbar.ProgressBar(max_value=n)

        for i in xrange(n):
            self.reset_objective()
            self._set_objective_from_data(
                self.data.sample(self.data.shape[0], replace=True),
                self.weights)
            self.initialize(print_level=0, print_time=0, **solve_opts)
            try:
                if solve_method is 'warm':
                    self.warm_solve()
                else:
                    self.solve()

                results += [map_fn(self)]

            except Exception:
                pass

            if progress: pbar.update(i)

        return results

    @property
    def rss(self):
        """ Residual sum of squares """
        
        x_reg = pd.DataFrame([
            self._get_interp(t, states=self.data.columns, x_rep='op')
            for t in self.data.index], index=self.data.index,
                             columns=self.data.columns)

        return ((self.data - x_reg)**2).sum().sum()

    @property
    def aic(self):
        """ Akaike information criterion """

        n = np.multiply(*self.data.shape)
        k = self.np

        return 2*k + n*np.log(self.rss)

    def _interpolate_solution(self, ts):

        h = self.tf / self.nk
        stage_starts = pd.Series(h * np.arange(self.nk))
        stages = stage_starts.searchsorted(ts, side='right') - 1

        out = np.empty((len(ts), self.nx))
    
        for ki in range(self.nk):
            for ni in range(self.nx):
                interp = lagrange(self.col_vars['tau_root'], 
                                  self.var.x_op[ki, :, ni])

                out[stages == ki, ni] = interp(
                    (ts[stages == ki] - stage_starts[ki])/h)

        return out
