import pandas as pd
import numpy as np

import casadi as cs
import cobra

from .VariableHandler import VariableHandler
from .BaseCollocation import BaseCollocation

class EFMcollocation(BaseCollocation):

    def __init__(self, model, boundary_species, efms):
        """ A class to handle the dynamic optimization based method of dynamic
        flux analysis from [1].

        model: a cobra.Model object.
            Contains the model to be optimized, reaction bounds and objective
            vectors will be preserved.

        boundary_species: a list
            Contains a list of strings specifying the boundary species which
            will be explicity tracked in the dynamic optimization. Biomass
            should be the first entry.

        efms: a pandas dataframe
            efms should be a dataframe containing compressed, boundary EFMs
            (with EFMs as columns) 


        """

        self.model = cobra.core.DataframeBasedModel(model)
        # self.model.optimize(minimize_absolute_flux=1.0)

        self.nx = len(boundary_species)
        self.nv = efms.shape[0]
        self.nm = len(self.model.metabolites)

        # If tf is left as None, it will be a symbolic variable
        self.tf = None

        assert self.nx == efms.shape[1], "EFMs are the wrong shape"
        
        # Handle boundary reactions
        self.boundary_species = boundary_species
        all_boundary_rxns = model.reactions.query('system_boundary', 'boundary')

        self.boundary_rxns = []
        for bs in boundary_species:
            rxns = all_boundary_rxns.query(lambda r: r.reactants[0].id in bs)

            assert len(rxns) == 1, (
                "Error finding boundary reactions for {}: ".format(bs) + 
                "{:d} reactions found".format(len(rxns)))

            self.boundary_rxns += [rxns[0].id]

        # Assure that efms are in the correct order
        self.efms = efms.loc[:, self.boundary_rxns]

        super(EFMcollocation, self).__init__()

    def setup(self, mav=False):
        """ Set up the collocation framework """

        self._initialize_dynamic_model()
        self._initialize_polynomial_coefs()
        self._initialize_variables()
        self._initialize_polynomial_constraints()

        self.col_vars['mav'] = mav

    def initialize(self, **kwargs):
        """ Call after setting up boundary kinetics, finalizes the
        initialization and sets up the NLP problem. Keyword arguments are
        passed directly as options to the NLP solver """

        if self.col_vars['mav']: self._initialize_mav_objective()
        self._initialize_solver(**kwargs)

    def plot_optimal(self):
        """ Method to quickly plot an optimal solution. """


        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.set_context('talk')

        fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1, figsize=(12,10))

        # Plot the results
        lines = ax[0].plot(self.ts, self.sol[:,1:], '.--')
        ax[0].legend(lines, self.boundary_species[1:], loc='upper left', ncol=2)
    
        # Plot the optimal fluxes
        self._plot_optimal_fluxes(ax[1])

        # Plot the biomass results
        lines = ax[2].plot(self.ts, self.sol[:,0], '.--')
        ax[2].legend(lines, self.boundary_species, loc='upper left', ncol=2)

        plt.show()


    def add_boundary_constraints(self, constraint_dictionary):
        """ Add dynamic constraints to flux across the system boundary in order
        to account for enzyme kinetics. 
        
        constraint_dictionary should be a dictionary of metabolite : function
        pairs, where 

            `metabolite` is in self.boundary_species, and

            `function` is a function taking a single argument, `x`, a
            dictionary containing symbolic representions of the current
            boundary concentrations, and return a tuple of (lb, ub) specifying
            the symbolic lower and upper bounds of the flux at that finite
            element. A `None` for either lb or ub will skip that constraint,
            using the constant bound specified from the model.

        """

        for key in constraint_dictionary.iterkeys():
            assert key in self.boundary_species, "{} not found".format(key)

        # Iterate over each finite element
        for k in xrange(self.nk):
            
            # Create a dictionary to pass to the bounds functions
            x = {met : var_sx for met, var_sx in 
                 zip(self.boundary_species, self.var.x_sx[k,0])}

            for met, boundary_func in constraint_dictionary.iteritems():
                rxn = self.boundary_rxns[self.boundary_species.index(met)]
                rxn_sx = self.efms.loc[:, rxn].dot(self.var.v_sx.loc[k])
                lb, ub = boundary_func(x)

                if lb is not None:
                    self.col_vars['v_lb_sym'].loc[k, rxn] = lb
                    # self.var.v_lb.loc[k, rxn] = -1000
                    self.add_constraint(rxn_sx - lb, 0, cs.inf)


                if ub is not None:
                    self.col_vars['v_ub_sym'].loc[k, rxn] = ub
                    self.add_constraint(ub - rxn_sx, 0, cs.inf)

    def _initialize_dynamic_model(self):
        """ Initialize the model of biomass growth and substrate accumulation
        """

        t  = cs.SX.sym('t')          # not really used.
        x  = cs.SX.sym('x', self.nx) # External metabolites
        vb = cs.SX.sym('v', self.nx) # fluxes across the boundary

        xdot = vb * x[0]             # Here we assume biomass is in x[0]

        self.dxdt = cs.SXFunction('dxdt', [t,x,vb], [xdot])

    def _initialize_variables(self):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'v'  : (self.nk, self.nv),
            # 'a'  : (self.nk, self.d+1),
        }

        if self.tf == None: core_variables.update({'tf' : (1,)})

        self.var = VariableHandler(core_variables, self.efms.index.values)

        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 100.

        # Initialize EFM bounds. EFMs are nonnegative.
        self.var.v_lb[:] = 0.
        self.var.v_ub[:] = cs.inf
        self.var.v_in[:] = 0.

        # Activity polynomial.
        # self.var.a_lb[:] = 0.
        # self.var.a_ub[:] = np.inf
        # self.var.a_in[:] = 1.

        if self.tf == None:
            self.var.tf_lb[:] = 1.
            self.var.tf_ub[:] = 100.
            self.var.tf_in[:] = 10.

        # Initialize symbolic upper and lower bound flux arrays to hold
        # boundary fluxes
        self.col_vars['v_lb_sym'] = pd.DataFrame(
            np.empty((self.nk, self.nx), dtype=object),
            columns=self.boundary_rxns)
        self.col_vars['v_ub_sym'] = pd.DataFrame(
            np.empty((self.nk, self.nx), dtype=object),
            columns=self.boundary_rxns)

    def _initialize_polynomial_constraints(self):
        """ Add constraints to the model to account for system dynamics and
        continuity constraints """

        if self.tf == None:
            h = self.var.tf_sx / self.nk
        else:
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
                    [T[k,j], cs.SX(self.var.x_sx[k,j]),
                     cs.SX(self.efms.T.dot(self.var.v_sx.loc[k]).values)])

                self.add_constraint(h*fk - xp_jk)

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

        self.objective_sx += (self.col_vars['alpha'] *
                              cs.fabs(self.var.v_sx).values.sum())


    def _plot_optimal_fluxes(self, ax):
        active_fluxes = self.var.v_op.max(0) > 1E-4
        ax.step(self.fs, self.var.v_op.loc[:, active_fluxes])

    def _plot_setup(self):

        # Create vectors from optimized time and states
        if self.tf == None:
            h = self.var.tf_op / self.nk
        else: 
            h = self.tf / self.nk

        self.fs = h * np.arange(self.nk)
        self.ts = np.array(
            [point + h*np.array(self.col_vars['tau_root']) for point in 
             np.linspace(0, h * self.nk, self.nk,
                         endpoint=False)]).flatten()

        self.sol = self.var.x_op.reshape((self.nk*(self.d+1)), self.nx)
        
        
        
        





        

    
    
