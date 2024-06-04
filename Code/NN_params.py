import scipy.io
import os

class NN_params:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path

        # Load all data
        self.load_data()

        # Define network size
        self.layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 2]

    def load_data(self):
        self.load_region_data()
        self.load_boundary_data()
        self.load_equation_parameters()
        self.load_boundaries()
        self.load_prior_data()

    def load_region_data(self):
        regions = ['Region1_Data.mat', 'Region2_Data.mat', 'Region3_Data.mat', 'Region4_Data.mat']
        y_coords = []
        x_coords = []

        for region in regions:
            file_path = os.path.join(self.data_path, region)
            data = scipy.io.loadmat(file_path)
            y_coords.append(data['y'])
            x_coords.append(data['x'])

        self.y_1, self.y_2, self.y_3, self.y_4 = y_coords
        self.x_1, self.x_2, self.x_3, self.x_4 = x_coords

    def load_boundary_data(self):
        boundary_data = scipy.io.loadmat(f'{self.data_path}/Boundaries_Data.mat')
        self.y_AB = boundary_data['y_AB']
        self.x_BC = boundary_data['x_BC']
        self.y_CD = boundary_data['y_CD']
        self.x_DE = boundary_data['x_DE']
        self.y_EF = boundary_data['y_EF']
        self.x_FG = boundary_data['x_FG']
        self.y_GH = boundary_data['y_GH']
        self.x_HI = boundary_data['x_HI']
        self.y_IJ = boundary_data['y_IJ']
        self.x_JK = boundary_data['x_JK']
        self.y_KL = boundary_data['y_KL']
        self.x_LA = boundary_data['x_LA']

    def load_equation_parameters(self):
        eq_params = scipy.io.loadmat(f'{self.data_path}/Equation_Parameter.mat')
        param_names = [
            'D11', 'D21', 'D12', 'D22', 'D13', 'D23', 'D14', 'D24',
            'Sigma121', 'Sigma122', 'Sigma123', 'Sigma124',
            'Sigmaa11', 'Sigmaa21', 'Sigmaa12', 'Sigmaa22',
            'Sigmaa13', 'Sigmaa23', 'Sigmaa14', 'Sigmaa24',
            'vSigmaf21', 'vSigmaf22', 'vSigmaf23', 'vSigmaf24',
            'Bz1', 'Bz2'
        ]

        for name in param_names:
            setattr(self, name, eq_params[name])

    def load_boundaries(self):
        lbub_data = scipy.io.loadmat(f'{self.data_path}/lower_upper_boundaries.mat')
        self.lb = lbub_data['lb']
        self.ub = lbub_data['ub']

    def load_prior_data(self):
        prior_data = scipy.io.loadmat(f'{self.data_path}/Train_Prior_Data.mat')
        self.x_prior = prior_data['x_prior']
        self.y_prior = prior_data['y_prior']
        self.u_prior = prior_data['u_prior']
        self.v_prior = prior_data['v_prior']

    def get_config_data(self):
        train_dict = {}
        train_variables = ['lb', 'ub', 'y_1', 'x_1', 'y_2', 'x_2', 'y_3', 'x_3', 'y_4', 'x_4',
                           'y_AB', 'x_BC', 'y_CD', 'x_DE', 'y_EF', 'x_FG', 'y_GH', 'x_HI', 'y_IJ',
                           'x_JK', 'y_KL', 'x_LA', 'x_prior', 'y_prior', 'u_prior', 'v_prior']

        for variable_name in train_variables:
            train_dict[variable_name] = getattr(self, variable_name)
        param_dict = {}
        param_variables = ['D11', 'D21', 'D12', 'D22', 'D13', 'D23', 'D14', 'D24',
                           'Sigma121', 'Sigma122', 'Sigma123', 'Sigma124',
                           'Sigmaa11', 'Sigmaa21', 'Sigmaa12', 'Sigmaa22',
                           'Sigmaa13', 'Sigmaa23', 'Sigmaa14', 'Sigmaa24',
                           'vSigmaf21', 'vSigmaf22', 'vSigmaf23', 'vSigmaf24',
                           'Bz1', 'Bz2', 'layers', 'data_path', 'device']

        for variable_name in param_variables:
            param_dict[variable_name] = getattr(self, variable_name)

        return train_dict, param_dict

