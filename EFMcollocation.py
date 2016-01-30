import pandas as pd
import numpy as np

import casadi as cs
import cobra

from .VariableHandler import VariableHandler
from .DOAcollocation import DOAcollocation
import warnings

class EFMcollocation(DOAcollocation):

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

        super(EFMcollocation, self).__init__(model, boundary_species)

        assert self.nx == efms.shape[1], "EFMs are the wrong shape"

        # Assure that efms are in the correct order
        self.efms = efms.loc[:, self.boundary_rxns]

        # Re-allocate for the EFM flux vector
        self.nv = efms.shape[0]

    def setup(self, mav=False):
        """ Set up the collocation framework """

        self._initialize_dynamic_model()
        self._initialize_polynomial_coefs()
        self._initialize_variables()
        self._initialize_polynomial_constraints()

        self.col_vars['mav'] = mav
        self.col_vars['kkt'] = False


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
                    self.constraints_sx.append(rxn_sx - lb)
                    self.constraints_ub.append(np.array([cs.inf]))
                    self.constraints_lb.append(np.array([0]))


                if ub is not None:
                    self.col_vars['v_ub_sym'].loc[k, rxn] = ub
                    self.constraints_sx.append(ub - rxn_sx)
                    self.constraints_ub.append(np.array([cs.inf]))
                    self.constraints_lb.append(np.array([0]))
                    # self.var.v_ub.loc[k, rxn] = +1000



    def _initialize_variables(self):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'v'  : (self.nk, self.nv),
        }

        if self.tf == None: core_variables.update({'tf' : (1,)})

        self.var = VariableHandler(core_variables, self.efms.index.values)

        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 100.

        # Initialize EFM bounds. EFMs are nonnegative.
        self.var.v_lb[:] = 0.
        self.var.v_ub[:] = np.inf
        self.var.v_in[:] = 0.

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



        
        
        
        





        

    
    
