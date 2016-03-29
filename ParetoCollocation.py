import numpy as np
import pandas as pd

from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumtrapz

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

    def solve_series(self, xs, reduce_function, warmsolve=True):

        xs = np.asarray(xs)
        xs.sort()

        # Container to hold output of reduce_function
        out = {}

        if progressbar:
            bar = progressbar.ProgressBar(max_value=len(xs))

        for i, x in enumerate(xs):
            try:
                if warmsolve:
                    if i == 0: self.solve(x)
                    else: self.warm_solve(x)
                else:
                    self.solve(x)

                out[x] = reduce_function(self)
            except RuntimeWarning:
                out[x] = np.nan

            if progressbar: bar.update(i)

        return pd.DataFrame(out).T


class AlphaInterpolator(object):

    def __init__(self, a, x, y):

        # Drop NaN values to avoid fitpack errors
        self._data = pd.DataFrame(np.array([a, x, y]).T, 
                                  columns=['a', 'x', 'y'])
        self._data.dropna(inplace=True)

        self._create_interpolating_polynomials()
        self._find_path_length()


    def _create_interpolating_polynomials(self):
        self.x_interp = UnivariateSpline(self._data.a, self._data.x, s=0)
        self.y_interp = UnivariateSpline(self._data.a, self._data.y, s=0)


    def _find_path_length(self):
        dx_interp = self.x_interp.derivative()
        dy_interp = self.y_interp.derivative()

        ts = np.linspace(0, 1, 200)
        line_length = cumtrapz(np.sqrt(dx_interp(ts)**2 + dy_interp(ts)**2),
                               x=ts, initial=0.) 

        line_length /= line_length.max()

        # Here we invert the line_length (ts) function, in order to evenly
        # sample the pareto front
        self.l_interp = UnivariateSpline(line_length, ts, s=0)

    def sample(self, num):
        """ Return estimates of alpha values that evenly sample the pareto
        front """
        
        out = self.l_interp(np.linspace(0, 1, num))
        out[0] = 0.
        out[-1] = 1.
        return out

