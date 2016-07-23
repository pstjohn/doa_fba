import pandas as pd
import numpy as np

import casadi as cs
from functools import reduce


class VariableHandler(object):

    def __init__(self, shape_dict):
        """ A class to handle the flattening and expanding of the NLP variable
        vector. Solves a lot of headaches.

        shape_dict: a dictionary containing variable_name : shape pairs, where
        shape is a tuple of the desired variable dimensions.
        
        """

        self._data = pd.DataFrame(pd.Series(shape_dict), columns=['shapes'])

        from operator import mul
        def elements(shape):
            try: return reduce(mul, shape)
            except TypeError: return shape

        self._data['lengths'] = self._data.shapes.apply(elements)
        self._data['end'] = self._data.lengths.cumsum()
        self._data['start'] = self._data.end - self._data.lengths
        self._total_length = self._data.lengths.sum()

        # Initialize symbolic variable
        self.vars_sx = cs.SX.sym('vars', self._total_length)

        # Split symbolic variable
        symbolic_dict = self._expand(self.vars_sx)

        for key, row in self._data.iterrows():
            self.__dict__.update({
                key + '_lb' : np.zeros(row.shapes), # Lower bounds
                key + '_ub' : np.zeros(row.shapes), # Upper bounds
                key + '_in' : np.zeros(row.shapes), # Initial guess
                key + '_op' : np.zeros(row.shapes), # Optimized Value
                key + '_sx' : symbolic_dict[key],
           })


    @property
    def vars_lb(self): return self._condense('lb')

    @vars_lb.setter
    def vars_lb(self, vars_lb):
        expanded = self._expand(vars_lb)
        for key, val in expanded.items():
            self.__dict__.update({key + '_lb' : val})


    @property
    def vars_ub(self): return self._condense('ub')

    @vars_ub.setter
    def vars_ub(self, vars_ub):
        expanded = self._expand(vars_ub)
        for key, val in expanded.items():
            self.__dict__.update({key + '_ub' : val})


    @property
    def vars_in(self): return self._condense('in')

    @vars_in.setter
    def vars_in(self, vars_in):
        expanded = self._expand(vars_in)
        for key, val in expanded.items():
            self.__dict__.update({key + '_in' : val})


    @property
    def vars_op(self): return self._condense('op')

    @vars_op.setter
    def vars_op(self, vars_op):
        expanded = self._expand(vars_op)
        for key, val in expanded.items():
            self.__dict__.update({key + '_op' : val})


    def _condense(self, suffix):
        """ Flatten the given variables to give a single dimensional vector """

        vector_out = np.zeros(self._total_length)
        for key, row in self._data.iterrows():
            try:
                vector_out[row.start:row.end] = \
                    self.__dict__[key + '_' + suffix].flatten()
            except AttributeError:
                # Allow for pandas dataframes
                vector_out[row.start:row.end] = \
                    self.__dict__[key + '_' + suffix].values.flatten()

        return vector_out


    def _expand(self, vector):
        """ Given a flattened vector, expand into the component matricies """

        # vector = np.asarray(vector)
        vector = np.array([vector[i] for i in range(vector.shape[0])])
        assert len(vector) == self._total_length, "expand length mismatch"

        def reshape_slice(row, key):
            return vector[row.start:row.end].reshape(row.shapes)

        return {key : reshape_slice(row, key) for key, row in
                self._data.iterrows()}

        return pd.Series([reshape_slice(row, key) for key, row in
                          self._data.iterrows()], index = self._data.index)

    def __getstate__(self):
        result = self.__dict__.copy()
        to_delete = [key for key in result.keys() if key.endswith('_sx')]
        for key in to_delete:
            del result[key]
        return result
    
    def __setstate__(self, result):
        self.__dict__ = result
        self.vars_sx = cs.SX.sym('vars', self._total_length)
        symbolic_dict = self._expand(self.vars_sx)
        for key, row in self._data.iterrows():
            self.__dict__.update({
                key + '_sx' : symbolic_dict[key],
            }) 
