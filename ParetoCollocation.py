import numpy as np
import pandas as pd
from progressbar import ProgressBar

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

    def solve_series(self, xs, reduce_function):

        xs = np.asarray(xs)
        xs.sort()

        # Container to hold output of reduce_function
        out = {}

        with ProgressBar(max_value=len(xs)) as bar:
            for i, x in enumerate(xs):
                if i == 0: self.solve(x)
                else: self.warm_solve(x)
                out[x] = reduce_function(self)
                bar.update(i)

        return pd.DataFrame(out).T
