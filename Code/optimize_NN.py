import torch
from build_model import PhysicsInformedNN
from NN_params import NN_params
import scipy.io


class OptimConfig:
    def __init__(self, device, data_path):
        self.device = device
        self.data_path = data_path

        # Initialize training configuration and model
        train_config = NN_params(self.device, self.data_path)
        self.train_data, self.param_data = train_config.get_config_data()

        # Create and configure the Physics-Informed Neural Network model
        self.model = PhysicsInformedNN(train_dict=self.train_data, param_dict=self.param_data)
        self.model.to(self.device)

    def optimize_and_predict(self, adam_steps, lbfgs_steps):
        # Configure the Adam optimizer
        adam_optimizer = torch.optim.Adam(
            params=self.model.weights + self.model.biases + [self.model.lambda_1],
            lr=1e-3
        )
        self.model.train_Adam(adam_optimizer, adam_steps, None)

        # Configure the L-BFGS optimizer
        lbfgs_optimizer = torch.optim.LBFGS(
            params=self.model.weights + self.model.biases + [self.model.lambda_1],
            lr=1,
            max_iter=lbfgs_steps
        )
        self.model.train_LBFGS(lbfgs_optimizer, None)

        # Load prediction data
        prediction_data = scipy.io.loadmat(self.data_path + '/2DIBP_X_Pred.mat')
        self.x_pred = prediction_data['x_pred']
        self.y_pred = prediction_data['y_pred']

        # Generate predictions using the trained model
        u_pred, v_pred = self.model.predict(self.x_pred, self.y_pred)

        # Save the predictions to a MATLAB file
        scipy.io.savemat(self.data_path + '/2DIBP_U_Pred.mat', {'u_pred': u_pred, 'v_pred': v_pred})

        # Plot and save the loss and keff graphs
        self.model.plot_loss(self.data_path)
        self.model.plot_keff(self.data_path)