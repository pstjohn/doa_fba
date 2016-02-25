import numpy as np

import casadi as cs


class BaseCollocation(object):

    def __init__(self):

        # Set up defaults for these collocation parameters (can be re-assigned
        # prior to collocation initialization
        self.nk = 20
        self.d = 2
        
        # Initialize container variables
        self.col_vars = {}
        self._constraints_sx = []
        self._constraints_lb = []
        self._constraints_ub = []
        self.objective_sx = 0.

    def add_constraint(self, sx, lb=None, ub=None):
        """ Add a constraint to the problem. sx should be a casadi symbolic
        variable, lb and ub should be the same length. If not given, upper and
        lower bounds default to 0. 

        Replaces manual addition of constraint variables to allow for warnings
        to be issues when a constraint that returns 'nan' with the current
        initalized variables is added.

        """

        constraint_len = sx.shape[0]
        assert sx.shape[1] == 1, "SX shape {} mismatch".format(sx.shape)

        if lb is None: lb = np.zeros(constraint_len)
        else: lb = np.atleast_1d(np.asarray(lb))

        if ub is None: ub = np.zeros(constraint_len)
        else: ub = np.atleast_1d(np.asarray(ub))

        # Make sure the bounds are sensible
        assert len(lb) == constraint_len, "LB length mismatch"
        assert len(ub) == constraint_len, "UB length mismatch"
        assert np.all(lb <= ub), "LB ! <= UB"

        try:
            gfcn = cs.SXFunction('g test',
                                 [self.var.vars_sx, self.pvar.vars_sx],
                                 [sx])
            out = np.asarray(gfcn([self.var.vars_in, 1.])[0])
            if np.any(np.isnan(out)):
                raise RuntimeWarning('Constraint gives NAN with default input '
                                     'arguments')
        
        except (AttributeError, KeyError):
            pass

        self._constraints_sx.append(sx)
        self._constraints_lb.append(lb)
        self._constraints_ub.append(ub)



    def solve(self):
        """ Solve the NLP. Alpha specifies the value for the regularization
        parameter, which minimizes the sum |v|.

        """
        
        # Fill argument call dictionary
        arg = {
            'x0'  : self.var.vars_in,
            'lbx' : self.var.vars_lb,
            'ubx' : self.var.vars_ub,

            'lbg' : self.col_vars['lbg'],
            'ubg' : self.col_vars['ubg'],

            'p'   : self.pvar.vars_in,
        }


        # Call the solver
        self._result = self._solver(arg)

        # Process the optimal vector
        self.var.vars_op = self._result['x']


        # Store the optimal solution as initial vectors for the next go-around
        self.var.vars_in = self.var.vars_op

        try: self._plot_setup()
        except AttributeError: pass

        return float(self._result['f'])


    def _initialize_polynomial_coefs(self):
        """ Setup radau polynomials and initialize the weight factor matricies
        """
        self.col_vars['tau_root'] = cs.collocationPoints(self.d, "radau")

        # Dimensionless time inside one control interval
        tau = cs.SX.sym("tau")

        # For all collocation points
        L = [[]]*(self.d+1)
        for j in range(self.d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the
            # collocation point
            L[j] = 1
            for r in range(self.d+1):
                if r != j:
                    L[j] *= (
                        (tau - self.col_vars['tau_root'][r]) / 
                        (self.col_vars['tau_root'][j] -
                         self.col_vars['tau_root'][r]))

        self.col_vars['lfcn'] = lfcn = cs.SXFunction(
            'lfcn', [tau], [cs.vertcat(L)])

        # Evaluate the polynomial at the final time to get the coefficients of
        # the continuity equation
        # Coefficients of the continuity equation
        self.col_vars['D'] = lfcn([1.0])[0].toArray().squeeze()

        # Evaluate the time derivative of the polynomial at all collocation
        # points to get the coefficients of the continuity equation
        tfcn = lfcn.tangent()

        # Coefficients of the collocation equation
        self.col_vars['C'] = np.zeros((self.d+1, self.d+1))
        for r in range(self.d+1):
            self.col_vars['C'][:,r] = tfcn([self.col_vars['tau_root'][r]]
                                           )[0].toArray().squeeze()

        # Find weights for gaussian quadrature: approximate int_0^1 f(x) by
        # Sum(
        xtau = cs.SX.sym("xtau")

        Phi = [[]] * (self.d+1)

        for j in range(self.d+1):
            tau_f_integrator = cs.SXFunction('ode', cs.daeIn(t=tau, x=xtau),
                                             cs.daeOut(ode=L[j]))
            tau_integrator = cs.Integrator(
                "integrator", "cvodes", tau_f_integrator, {'t0':0., 'tf':1})
            Phi[j] = np.asarray(tau_integrator({'x0' : 0})['xf'])[0][0]

        self.col_vars['Phi'] = np.array(Phi)
        
    def _initialize_solver(self, **kwargs):

        # Initialize NLP object
        self._nlp = cs.SXFunction(
            'nlp', 
            cs.nlpIn(x = self.var.vars_sx,
                     p = self.pvar.vars_sx),
            cs.nlpOut(f = self.objective_sx, 
                      g = cs.vertcat(self._constraints_sx)))

        opts = {
            'max_iter' : 10000,
            'linear_solver' : 'ma27'
        }
        
        if kwargs is not None: opts.update(kwargs)

        self._solver_opts = opts

        self._solver = cs.NlpSolver("solver", "ipopt", self._nlp,
                                    self._solver_opts)

        self.col_vars['lbg'] = np.concatenate(self._constraints_lb)
        self.col_vars['ubg'] = np.concatenate(self._constraints_ub)

    def warm_solve(self, x0=None, lam_x=None, lam_g=None):
        """Solve the collocation problem using an initial guess and basis from
        a prior solve. Defaults to using the variables from the solve stored in
        _results. 

        """
        warm_solve_opts = dict(self._solver_opts)

        warm_solve_opts["warm_start_init_point"] = "yes"
        warm_solve_opts["warm_start_bound_push"] = 1e-6
        warm_solve_opts["warm_start_slack_bound_push"] = 1e-6
        warm_solve_opts["warm_start_mult_bound_push"] = 1e-6

        solver = self._solver = cs.NlpSolver("solver", "ipopt", self._nlp,
                                             warm_solve_opts)


        if x0 is None: x0 = self._result['x']
        if lam_x is None: lam_x = self._result['lam_x']
        if lam_g is None: lam_g = self._result['lam_g']

        solver.setInput(x0, 'x0')
        solver.setInput(self.var.vars_lb, 'lbx')
        solver.setInput(self.var.vars_ub, 'ubx')
        solver.setInput(self.col_vars['lbg'], 'lbg')
        solver.setInput(self.col_vars['ubg'], 'ubg')
        solver.setInput(self.pvar.vars_in, 'p')
        solver.setInput(lam_x, 'lam_x0')
        solver.setInput(lam_g, 'lam_g0')
        solver.setOutput(lam_x, "lam_x")

        self._solver.evaluate()
        self._result = {
            'x' : self._solver.getOutput('x'),
            'lam_x' : self._solver.getOutput('lam_x'),
            'lam_g' : self._solver.getOutput('lam_g'),
            'f' : self._solver.getOutput('f'),
        }

        # Process the optimal vector
        self.var.vars_op = self._result['x']

        # Store the optimal solution as initial vectors for the next go-around
        self.var.vars_in = self.var.vars_op

        try: self._plot_setup()
        except AttributeError: pass

        return float(self._result['f'])
