import pandas as pd
import numpy as np

import casadi as cs
import cobra

from .VariableHandler import VariableHandler
from .BaseCollocation import BaseCollocation

class DOAcollocation(BaseCollocation):

    def __init__(self, model, boundary_species, add_biomass=False):
        """ A class to handle the dynamic optimization based method of dynamic
        flux analysis from [1].

        model: a cobra.Model object.
            Contains the model to be optimized, reaction bounds and objective
            vectors will be preserved.

        boundary_species: a list
            Contains a list of strings specifying the boundary species which
            will be explicity tracked in the dynamic optimization. Biomass
            should be the first entry.

        add_biomass: bool
            Whether or not to attempt to add a `biomass` metabolite to the
            model. If True, will add the metabolite to the reaction specified
            by the objective function.

        ---
        [1] J. L. Hjersted and M. A. Henson, "Optimization of
        Fed-Batch Saccharomyces cerevisiae Fermentation Using Dynamic Flux
        Balance Models," Biotechnol Progress, vol. 22, no. 5, pp. 1239-1248,
        Sep. 2008. 
        """

        if add_biomass:

            biomass = cobra.Metabolite('biomass', compartment='e')
            biomass_rxn = model.objective.keys()[0]
            biomass_rxn.add_metabolites({biomass : 1})
            EX_biomass = cobra.Reaction('EX_biomass')
            EX_biomass.add_metabolites({biomass : -1})
            model.add_reaction(EX_biomass)
            

        self.model = cobra.core.DataframeBasedModel(model)
        self.model.optimize(minimize_absolute_flux=1.0)

        
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

        self.nx = len(boundary_species)
        self.nv = len(self.model.reactions)
        self.nm = len(self.model.metabolites)

        # If tf is left as None, it will be a symbolic variable
        self.tf = None
        self.death_rate = None

        # Initialize the base class
        super(DOAcollocation, self).__init__()



    def setup(self, kkt=False, mav=False):
        """ Set up the collocation framework """

        self._initialize_dynamic_model()
        self._initialize_polynomial_coefs()
        self._initialize_variables(kkt)
        self._initialize_polynomial_constraints()
        self._initialize_mass_balance()

        self.col_vars['kkt'] = kkt
        self.col_vars['mav'] = mav


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
                rxn_sx = self.var.v_sx.loc[k, rxn]
                lb, ub = boundary_func(x)

                if lb is not None:
                    self.col_vars['v_lb_sym'].loc[k, rxn] = lb
                    self.var.v_lb.loc[k, rxn] = -1000
                    self.constraints_sx.append(rxn_sx - lb)
                    self.constraints_ub.append(np.array([cs.inf]))
                    self.constraints_lb.append(np.array([0]))


                if ub is not None:
                    self.col_vars['v_ub_sym'].loc[k, rxn] = ub
                    self.constraints_sx.append(ub - rxn_sx)
                    self.constraints_ub.append(np.array([cs.inf]))
                    self.constraints_lb.append(np.array([0]))
                    self.var.v_ub.loc[k, rxn] = +1000


    def initialize(self, **kwargs):
        """ Call after setting up boundary kinetics, finalizes the
        initialization and sets up the NLP problem. Keyword arguments are
        passed directly as options to the NLP solver """

        if self.col_vars['kkt']: self._initialize_kkt_constraints()
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

    def _plot_optimal_fluxes(self, ax):
        active_fluxes = self.var.v_op.max(0) > 1E-4
        ax.step(self.fs, self.var.v_op.loc[:, active_fluxes])


    def _initialize_dynamic_model(self):
        """ Initialize the model of biomass growth and substrate accumulation
        """

        t  = cs.SX.sym('t')          # not really used.
        x  = cs.SX.sym('x', self.nx) # External metabolites
        vb = cs.SX.sym('v', self.nx) # fluxes across the boundary

        xdot = vb * x[0]             # Here we assume biomass is in x[0]

        if self.death_rate is not None:
            xdot[0] -= self.death_rate * x[0]

        self.dxdt = cs.SXFunction('dxdt', [t,x,vb], [xdot])

        
    def _initialize_variables(self, kkt):
        """ Initialize the vector of unknowns """
         

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'v'  : (self.nk, self.nv),
        }

        if self.tf == None: core_variables.update({'tf' : (1,)})

        kkt_variables = {
            'Lambda' : (self.nk, self.nm),
            'alphaL' : (self.nk, self.nv),
            'alphaU' : (self.nk, self.nv),
            'etaL'   : (self.nk, self.nm),
            'etaU'   : (self.nk, self.nm),
        }

        if kkt: core_variables.update(kkt_variables)

        self.var = VariableHandler(core_variables, self.model.S.columns)

        # Initialize default variable bounds
        self.var.x_ub[:] = 100.

        # Initialize flux bounds to their model equivalents
        self.var.v_lb[:] = self.model.lower_bounds.values[
            :,np.newaxis].repeat(self.nk, axis=1).T
        self.var.v_ub[:] = self.model.upper_bounds.values[
            :,np.newaxis].repeat(self.nk, axis=1).T
        self.var.v_in[:] = self.model.fluxes.values[
            :,np.newaxis].repeat(self.nk, axis=1).T # Or 0?

        if self.tf == None:
            self.var.tf_lb[:] = 1.
            self.var.tf_ub[:] = 100.
            self.var.tf_in[:] = 10.

        # Recast flux array to dataframe for easier indexing
        # self.var.v_sx = pd.DataFrame(
        #     self.var.v_sx, columns=self.model.S.columns)
        # self.var.v_lb = pd.DataFrame(
        #     self.var.v_lb, columns=self.model.S.columns)
        # self.var.v_ub = pd.DataFrame(
        #     self.var.v_ub, columns=self.model.S.columns)

        # Initialize symbolic upper and lower bound flux arrays
        self.col_vars['v_lb_sym'] = pd.DataFrame(self.var.v_lb, dtype=object)
        self.col_vars['v_ub_sym'] = pd.DataFrame(self.var.v_ub, dtype=object)
        
        # Initialize KKT bounds
        if kkt:
            self.var.Lambda_lb[:] = -cs.inf
            self.var.Lambda_ub[:] = cs.inf
            self.var.Lambda_in[:] = 0.

            self.var.alphaL_lb[:] = 0
            self.var.alphaL_ub[:] = cs.inf
            self.var.alphaL_in[:] = 0.

            self.var.alphaU_lb[:] = 0
            self.var.alphaU_ub[:] = cs.inf
            self.var.alphaU_in[:] = 0.

            self.var.etaL_lb[:] =   0
            self.var.etaL_ub[:] =   cs.inf
            self.var.etaL_in[:] =   0.

            self.var.etaU_lb[:] =   0
            self.var.etaU_ub[:] =   cs.inf
            self.var.etaU_in[:] =   0.


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
                # (Pull boundary fluxes for this FE from the flux DF)
                [fk] = self.dxdt.call(
                    [T[k,j], cs.SX(self.var.x_sx[k,j]),
                     cs.SX(self.var.v_sx.loc[k, self.boundary_rxns].values)])

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
        



    def _initialize_mass_balance(self):
        """ Add mass balance constraints to ensure S.dot(v) == 0 at each finite
        element """

        mass_balance = self.model.S.dot(self.var.v_sx.T).values.flatten()
        self.constraints_sx += mass_balance.tolist()
        self.constraints_lb.append(np.zeros(self.nk * self.nm))
        self.constraints_ub.append(np.zeros(self.nk * self.nm))



    def _initialize_kkt_constraints(self):
        """ Initialize first-order optimality conditions for the flux
        distributions at each finite element. This will ensure the the
        objective function in the passed model object is respected, but adds
        significant computational cost. 
        
        Note: Must be run following the addition of custom boundary constraints
        """

        # KKT first-order optimality constraints
        kkt_mul1 = (self.model.objectives + 
                    self.model.S.T.dot(self.var.Lambda_sx.T).T + 
                    self.var.alphaL_sx - self.var.alphaU_sx)
        self.constraints_sx += kkt_mul1.values.flatten().tolist()
        self.constraints_lb.append(np.zeros(self.nk * self.nv))
        self.constraints_ub.append(np.zeros(self.nk * self.nv))

        kkt_mul2 = self.var.Lambda_sx - self.var.etaL_sx + self.var.etaU_sx
        self.constraints_sx += kkt_mul2.flatten().tolist()
        self.constraints_lb.append(np.zeros(self.nk * self.nm))
        self.constraints_ub.append(np.zeros(self.nk * self.nm))

        # Complimentarity constraints
        compL = ((self.var.v_sx - self.col_vars['v_lb_sym']) *
                 self.var.alphaL_sx)
        self.constraints_sx += compL.values.flatten().tolist()
        self.constraints_lb.append(np.zeros(self.nk * self.nv))
        self.constraints_ub.append(np.zeros(self.nk * self.nv))

        compU = ((self.col_vars['v_ub_sym'] - self.var.v_sx) *
                 self.var.alphaU_sx)
        self.constraints_sx += compU.values.flatten().tolist()
        self.constraints_lb.append(np.zeros(self.nk * self.nv))
        self.constraints_ub.append(np.zeros(self.nk * self.nv))


    def _initialize_mav_objective(self):
        """ Initialize the objective function to minimize the absolute value of
        the flux vector """

        self.objective_sx += (self.col_vars['alpha'] *
                              cs.fabs(self.var.v_sx).values.sum())

    
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








