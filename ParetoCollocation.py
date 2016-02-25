import numpy as np

from .VariableHandler import VariableHandler
from .EFMcollocation import EFMcollocation

class ParetoCollocation(EFMcollocation):
    def _initialize_variables(self):
        super(ParetoCollocation, self)._initialize_variables()

        parameters = {
            'alpha' : (1,),
            'c0'    : (1,),
            'c1'    : (1,),
        }

        self.pvar = VariableHandler(parameters)
        self.pvar.alpha_in[:] = 0.
        self.pvar.c0_in[:] = 1.
        self.pvar.c1_in[:] = 1.
        
    def solve(self, alpha):

        self.pvar.alpha_in[:] = alpha
        return super(ParetoCollocation, self).solve()
    
    def warm_solve(self, alpha):

        self.pvar.alpha_in[:] = alpha
        return super(ParetoCollocation, self).warm_solve()

    def calibrate_objectives(self):

        self.pvar.vars_in[:] = 1.

        c0 = np.abs(self.solve(0.))
        c1 = np.abs(self.solve(1.))

        self.pvar.c0_in[:] = 1./c0
        self.pvar.c1_in[:] = 1./c1



    
