# A class for parameter estimation of dynamic systems using orthogonal
# collocation on finite elements. Makes heavy use of the casADi package and
# examples.

from warnings import warn

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import casadi as cs


class Collocation(object):

    def __init__(self, model, state_names):
        """ Initialize the collocation object.

        model: a cs.SXFunction object
            a model that describes substrate uptake and biomass formation
            kinetics. Inputs should be [t, x, p], outputs should be [ode]

        state_names: list
            List of string ID's for each of the states in the model. The first
            state should represent the current biomass concentration.

        """
        
        # Assign sizing variables
        self.NEQ = model.getInput(1).shape[0]
        self.NP  = model.getInput(2).shape[0]

        assert model.getOutput(0).shape[0] == self.NEQ, \
            "Output length mismatch"

        # Attach model
        self.model = model

        # Attach state names
        assert len(state_names) == self.NEQ, "Name length mismatch"
        self.state_names = np.asarray(state_names)

    
    def initialize(self, opts=None):
        """ Initialize the algorithm for orthogonal collocation on finite
        elements by finding symbolic expressions for the embedded ODE calls and
        continuity equations. 
        
        Options should be passed through the optional 'opts' argument.
        Available options are:

            nk         : number of finite elements
            tf         : maximum time horizon
            x_min      : Minimum of state vector over trajectory (float or
                         array (NEQ))
            x_max      : Maximum of state vector over trajectory
            x0_min     : Minimum of initial state vector
            x0_max     : Maximum of initial state vector
            x_init     : Initial values for x if data is missing
            p_min      : Minimum of parameter vector
            p_max      : Maximum of parameter vector
            p_init     : Initial parameter guess if missing
            degree     : Degree of interpolating polynomial
            polynomial : Type of interpolating polynomial. "radau" or
                         "legendre"

        """

        # Initialize default options
        default_opts = {
            'nk'         : 20,
            'tf'         : 100.0,
            'x_min'      : 0.,
            'x_max'      : 100.,
            'x0_min'     : 0,
            'x0_max'     : 100.,
            'x_init'     : 0.1,
            'p_min'      : 0.,
            'p_max'      : 100.,
            'p_init'     : 0.,
            'degree'     : 3,
            'polynomial' : "radau",
        }

        # Overwrite default options with user-provided dictionary
        self.opts = {}
        self.opts.update(default_opts)
        # NOTE: The opts dictionary will also be used to hold various
        # problem-relevant constructs in order to avoid cluttering the class
        # namespace.
        
        if opts:
            for key, val in opts.items(): self.opts[key] = val

        self._initialize_tgrid()
        self._initialize_variables()
        self._initialize_continuity_constraints()
        self._initialize_variable_bounds()


    def _initialize_tgrid(self):
        # Choose collocation points
        tau_root = cs.collocationPoints(self.opts['degree'],
                                        self.opts['polynomial'])

        # Degree of interpolating polynomial
        d = self.opts['degree']

        # Size of the finite elements
        self.opts['h'] = self.opts['tf']/self.opts['nk']

        # Coefficients of the collocation equation
        self.opts['C'] = np.zeros((d+1,d+1))

        # Dimensionless time inside one control interval
        tau = cs.SX.sym("tau")

        T = np.zeros((self.opts['nk'], d+1))
        for k in range(self.opts['nk']):
            for j in range(d+1):
                T[k,j] = self.opts['h'] * (k + tau_root[j])
        self.opts['T'] = T


        # Construct Lagrange polynomials to get the polynomial basis at the
        # collocation point
        L = [[]]*(d+1)

        # For all collocation points
        for j in range(d+1):
            L[j] = 1
            for r in range(d+1):
                if r != j:
                    L[j] *= (tau - tau_root[r])/(tau_root[j] - tau_root[r])

        self._lfcn = cs.SXFunction('lfcn', [tau], [cs.vertcat(L)])

        # Evaluate the polynomial at the final time to get the coefficients of
        # the continuity equation
        self.opts['D'] = np.asarray(self._lfcn([1.0])[0]).squeeze()
        
        # Evaluate the time derivative of the polynomial at all collocation
        # points to get the coefficients of the continuity equation
        tfcn = self._lfcn.tangent()
        for r in range(d+1):
            self.opts['C'][:,r] = tfcn([tau_root[r]])[0].toArray().squeeze() 

        self._tgrid = np.array(
            [point + self.opts['h']*np.array(tau_root) for point in 
             np.linspace(0, self.opts['tf'], self.opts['nk'],
                         endpoint=False)]).flatten()


    def _initialize_variables(self):
        d = self.opts['degree']
        # Allocate symbolic variable array
        self.NV = self.opts['nk'] * (d + 1) * self.NEQ # Collocated points
        self.NV += self.NP                             # Parameters

        self._V = V = cs.SX.sym("V", self.NV)

        # Reshape variable vector into indexable state collocation array
        self._X = np.asarray(V)[:-self.NP].reshape(
            (self.opts['nk'], d+1, self.NEQ))


        # Symbolic parameters and bounds
        self._P = V[-self.NP:]

    def _initialize_continuity_constraints(self):
        # Constraint function for the NLP
        
        d = self.opts['degree']
        X = self._X
        P = self._P
        T = self.opts['T']

        g = []
        lbg = []
        ubg = []

        # For all finite elements
        for k in range(self.opts['nk']):

            # For all collocation points
            for j in range(1,d+1):

                # Get an expression for the state derivative at the collocation
                # point
                xp_jk = 0
                for r in range (d+1):
                    xp_jk += self.opts['C'][r,j]*cs.SX(X[k,r])

                # Add collocation equations to the NLP
                [fk] = self.model.call([T[k,j], cs.SX(X[k,j]), P])
                g.append(self.opts['h']*fk - xp_jk)
                lbg.append(np.zeros(self.NEQ)) # equality constraints
                ubg.append(np.zeros(self.NEQ)) # equality constraints

            # Add continuity equation to NLP
            if k+1 != self.opts['nk']:

                # Get an expression for the state at the end of the finite
                # element
                xf_k = 0
                for r in range(d+1):
                    xf_k += self.opts['D'][r] * cs.SX(X[k,r])

                g.append(cs.SX(X[k+1,0]) - xf_k)
                lbg.append(np.zeros(self.NEQ))
                ubg.append(np.zeros(self.NEQ))

        # Concatenate constraints
        self._g = cs.vertcat(g)
        self.opts['lbg'] = np.concatenate(lbg)
        self.opts['ubg'] = np.concatenate(ubg)


    def _initialize_variable_bounds(self):
        # Additional setup variables
        self.opts['vars_init'] = np.zeros(self.NV)
        self.opts['vars_lb'] = np.zeros(self.NV)
        self.opts['vars_ub'] = np.zeros(self.NV)


        # fill variable bounds
        self.opts['vars_lb'][-self.NP:] = self.opts['p_min']
        self.opts['vars_ub'][-self.NP:] = self.opts['p_max']

        # fill state bounds
        x_min = np.empty((self.opts['nk'], self.opts['degree'] + 1, self.NEQ))
        x_max = np.empty((self.opts['nk'], self.opts['degree'] + 1, self.NEQ))
        x_min[:,:,:] = self.opts['x_min']
        x_max[:,:,:] = self.opts['x_max']
        x_min[0,0,:] = self.opts['x0_min']
        x_max[0,0,:] = self.opts['x0_max']

        self.opts['vars_lb'][:-self.NP] = x_min.flatten()
        self.opts['vars_ub'][:-self.NP] = x_max.flatten()



    def _get_interp(self, t, states=None):
        """ Return a symbolic polynomial representation of the state vector
        evaluated at time t.

        states: list
            indicies of which states to return

        """

        if not states: states = range(1, self.NEQ)

        finite_element = int(t / self.opts['h'])
        tau = (t % self.opts['h']) / self.opts['h']
        basis = self._lfcn([tau])[0].toArray().flatten()
        x_roots = self._X[finite_element, :, states]

        return np.inner(basis, x_roots)


    def set_data(self, data):
        """ Attach experimental measurement data.

        data : a pd.DataFrame object
            Data should have columns corresponding to the state labels in
            self.state_names, with an index corresponding to the measurement
            times.

        """

        # Should raise an error if no state name is present
        df = data.loc[:, self.state_names]

        # Rename columns with state indicies
        df.columns = np.arange(self.NEQ)

        # Remove empty (nonmeasured) states
        self.data = df.loc[:, ~pd.isnull(df).all(0)]

        obj_list = []
        for ((ti, state), xi) in self.data.stack().items():
            obj_list += [(self._get_interp(ti, [state]) - xi) /
                         self.data[state].max()]

        obj_resid = cs.sum_square(cs.vertcat(obj_list))

        # ts = data.index

        # Define objective function
        # obj_resid = cs.sum_square(cs.vertcat(
        #     [(self._get_interp(ti, self._states_indicies) - xi)/ xs.max(0) 
        #      for ti, xi in zip(ts, xs)]))

        alpha = cs.SX.sym('alpha')
        obj_lasso = alpha * cs.sumRows(cs.fabs(self._P))

        self._obj = obj_resid + obj_lasso

        # Create the solver object
        self._nlp = cs.SXFunction('nlp', cs.nlpIn(x=self._V, p=alpha),
                                  cs.nlpOut(f=self._obj, g=self._g))


    def _estimate_initial_guess(self):
        """ Interpolate the given data to generate initial guesses for the
        collocation variables.
        
        """

        # Create data interpolation object
        data_interpolator = interp1d(self.data.index, self.data.values,
                                     kind='quadratic', axis=0)

        # Create a new dataframe to hold in initialized values
        ts = self.data.index
        in_range = (self._tgrid >= ts.min()) & (self._tgrid <= ts.max())
        reindexed = pd.DataFrame(index=pd.Index(self._tgrid),
                                 columns=self.state_names[self.data.columns])

        # Interpolate data
        reindexed.loc[reindexed.index < ts.min()] = self.data.values.min()
        reindexed.loc[in_range] = data_interpolator(self._tgrid[in_range])
        reindexed.loc[reindexed.index > ts.max()] = self.data.values.max()

        # reshape to problem dimensions
        reshaped = reindexed.loc[:, self.state_names]
        reshaped.fillna(self.opts['x_init'], inplace=True)

        self.opts['vars_init'][:-self.NP] = reshaped.values.flatten()




    def create_nlp(self, solve_opts=None):

        default_solve_opts = {
            'linear_solver' : 'ma27',
            'print_level'   : 0,
            'print_time'    : 0,
        }

        self._solve_opts = {}
        self._solve_opts.update(default_solve_opts)
        if solve_opts:
            for key, val in solve_opts.items(): self._solve_opts[key] = val

        # Initialize arguments for solve method

        self._nlp_solver = cs.NlpSolver("solver", "ipopt", self._nlp,
                                        self._solve_opts)
        

    def solve(self, args=None, param_guess=None, alpha=None, ode=True):
        """ Solve the NLP """

        # Fill p_inits
        if param_guess == None: param_guess = self.opts['p_init']
        self.opts['vars_init'][-self.NP:] = param_guess

        # Fill predicted x_inits
        self._estimate_initial_guess()

        # handle regularization parameter
        if alpha == None: alpha = 0.

        self._args = {
            'x0'  : self.opts['vars_init'],
            'lbx' : self.opts['vars_lb'],
            'ubx' : self.opts['vars_ub'],
            'lbg' : self.opts['lbg'],
            'ubg' : self.opts['ubg'],
            'p'   : alpha,
        }

        if args: 
            for key, val in args.items(): self._args[key] = val

        res = self._nlp_solver(self._args)
        self._output = self._parse_solver_output(res)

        if ode: self.solve_ode()

        return self._output

    def _parse_solver_output(self, res):
        """ parse NLP variable output """

        output = {}
        self.opts['vars_opt'] = v_opt = np.array(res["x"]).flatten()
        output['ts'] = self._tgrid
        output['x_opt'] = v_opt[:-self.NP].reshape(
            (self.opts['nk'] * (self.opts['degree'] + 1), self.NEQ))
        output['p_opt'] = v_opt[-self.NP:]
        output['f'] = res['f']
        
        return output

    

    def solve_ode(self):
        """ Solve the ODE using casadi's CVODES wrapper to ensure that the
        collocated dynamics match the error-controlled dynamics of the ODE """


        f_integrator = cs.SXFunction('ode',
                                     cs.daeIn(
                                         t = self.model.inputExpr(0),
                                         x = self.model.inputExpr(1),
                                         p = self.model.inputExpr(2)),
                                     cs.daeOut(
                                         ode = self.model.outputExpr(0)))

        integrator = cs.Integrator('int', 'cvodes', f_integrator)
        simulator = cs.Simulator('sim', integrator, self._tgrid)
        simulator.setInput(self._output['x_opt'][0], 'x0')
        simulator.setInput(self._output['p_opt'], 'p')
        simulator.evaluate()
        x_sim = self._output['x_sim'] = np.array(simulator.getOutput()).T

        err = ((self._output['x_opt'] - x_sim).mean(0) /
               (self._output['x_opt'].mean(0))).mean()

        if err > 1E-3: warn(
                'Collocation does not match ODE Solution: \
                {:.2f}% Error'.format(100*err))
        
    
    # def alpha_sweep(self, alphas):
    #     """ Solve the NLP over a range of different alpha values """

    #     if not hasattr(self, '_output'): self.solve()

    #     outputs = {}

    #     for alpha in alphas:
    #         self._args['x0'] = self.opts['vars_opt']
    #         res = self._nlp_solver(self._args)
    #         outputs[alpha] = self._parse_solver_output(res)

    #     return outputs






