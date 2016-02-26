import numpy as np
import pandas as pd

try:
    import progressbar
except ImportError:
    progressbar = False

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

        if progressbar:
            bar = progressbar.ProgressBar(max_value=len(xs))

        for i, x in enumerate(xs):
            try:
                if i == 0: self.solve(x)
                else: self.warm_solve(x)
                out[x] = reduce_function(self)
            except RuntimeWarning:
                out[x] = np.nan

            if progressbar: bar.update(i)

        return pd.DataFrame(out).T
