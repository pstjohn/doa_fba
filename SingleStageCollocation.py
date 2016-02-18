import pandas as pd
import numpy as np

import casadi as cs

from .VariableHandler import VariableHandler
from .EFMcollocation import EFMcollocation

class SingleStageCollocation(EFMcollocation):

    def _initialize_variables(self):

        core_variables = {
            'x'  : (self.nk, self.d+1, self.nx),
            'f'  : (self.nv), # Fractions of each EFM
            'a'  : (self.nk), # EFM activity at each FE
        }

        if self.tf == None: core_variables.update({'tf' : (1,)})

        self.var = VariableHandler(core_variables, self.efms.index.values)

        # Initialize default variable bounds
        self.var.x_lb[:] = 0.
        self.var.x_ub[:] = 100.

        # Initialize EFM bounds. EFMs are nonnegative.
        self.var.f_lb[:] = 0.
        self.var.f_ub[:] = 1.
        self.var.f_in[:] = 0.

        self.var.a_lb[:] = 0.
        self.var.a_ub[:] = 1E5
        self.var.a_in[:] = 0.

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

        
        # Add a fake variable to the vars framework that corresponds to the
        # scaled, chosen EFM. This will allow the methods of EFMcollocation to
        # work without the knowldege that EFMs don't actually vary from FE to
        # FE
        self.var.v_sx = pd.DataFrame(np.outer(self.var.a_sx, self.var.f_sx),
                                     columns=self.efms.index.values)


        # We also want the f_sx variable to represent a fraction of the overall
        # efm, so we'll add a constraint saying the sum of the variable must
        # equal 1.
        self.add_constraint(self.var.f_sx.sum(), 1., 1.,)


    def _plot_setup(self):

        # Similar to above, add in a fake "v_op" variable to fool the exisiting
        # superclass methods.
        self.var.v_op = pd.DataFrame(np.outer(self.var.a_op, self.var.f_op),
                                     columns=self.efms.index.values)
        
        super(EFMcollocation, self)._plot_setup()
        
    
