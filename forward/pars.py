class Par:

    def __init__(self, hidden_dim, batch_size, epochs, learn_rate, lr_decay):
        """
        define machine learning parameters
        """

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.test_portion = 0.1
        self.EPOCHS = epochs
        self.patience = 30  # early stopping patience; how long to wait after last time validation loss improved.
        self.dropout = 0.1
        self.learn_rate = learn_rate
        self.lr_decay = lr_decay


class Var_bicm:

    def __init__(self, x_vars, x_pars, y1_vars, y2_vars):
        self.x_vars = x_vars  # bicm_frcngs_vars
        self.x_pars = x_pars  # bicm_params_vars
        self.y_vars = y1_vars + y2_vars
        self.y1_vars = y1_vars  # bicm_obsrvs_var1
        self.y2_vars = y2_vars  # bicm_obsrvs_var2

        self.input_dim = len(self.x_vars + self.x_pars)
        self.output_dim = len(self.y_vars)
        self.output_dim1 = len(self.y1_vars)
        self.output_dim2 = len(self.y2_vars)


class Var_rtmo:

    def __init__(self, x_vars, x_pars, y1_vars, y2_vars):
        self.x_vars = x_vars  # rtmo_frcngs_vars
        self.x_pars = x_pars  # rtmo_params_vars
        self.y_vars = y1_vars + y2_vars
        self.y1_vars = y1_vars  # rtmo_obsrvs_var1
        self.y2_vars = y2_vars  # rtmo_obsrvs_var2

        self.input_dim = len(self.x_vars + self.x_pars)
        self.output_dim = len(self.y_vars)
        self.output_dim1 = len(self.y1_vars)
        self.output_dim2 = len(self.y2_vars)


class Var_carp:

    def __init__(self, x_vars, c_vars, x_pars, y_vars):
        self.x_vars = x_vars  # carp_frcngs_vars
        self.c_vars = c_vars  # carp_initls_vars
        self.x_pars = x_pars  # carp_params_vars
        self.y_vars = y_vars  # carp_obsrvs_vars

        self.input_dim = len(self.x_vars + self.c_vars + self.x_pars)
        self.output_dim = len(self.y_vars)
