from NN_params import NN_params
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, autograd
from torch.autograd import Variable

matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PINN(nn.Module):
    def __init__(self, train_dict, param_dict):
        super(PINN, self).__init__()

        # Unzip training and parameter dictionaries
        self.train_data = self.extract_train_data(train_dict)
        self.param_data = self.extract_param_data(param_dict)

        # Initialize model parameters and tensors
        self.initialize_model()

    def extract_train_data(self, train_dict):
        """
        Extract and return training data from the dictionary.
        """
        keys = ['lb', 'ub', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'yAB',
                'xBC', 'yCD', 'xDE', 'yEF', 'xFG', 'yGH', 'xHI', 'yIJ', 'xJK',
                'yKL', 'xLA', 'xp', 'yp', 'up', 'vp']
        return tuple(train_dict[ey] for key in keys)

    def extract_param_data(self, param_dict):
        """k
        Extract and return parameter data from the dictionary.
        """
        keys = ['D11', 'D21', 'D12', 'D22', 'D13', 'D23', 'D14', 'D24', 'Sigma121',
                'Sigma122', 'Sigma123', 'Sigma124', 'Sigmaa11', 'Sigmaa21', 'Sigmaa12',
                'Sigmaa22', 'Sigmaa13', 'Sigmaa23', 'Sigmaa14', 'Sigmaa24', 'vSigmaf21',
                'vSigmaf22', 'vSigmaf23', 'vSigmaf24', 'Bz1', 'Bz2', 'layers',
                'data_path', 'device']
        extracted_params = []

        for key in keys:
            if key in param_dict:
                extracted_params.append(param_dict[key])
            else:
                extracted_params.append(None)  # Append None for missing keys
        return tuple(extracted_params)

    def initialize_model(self):
        """
        Initialize model parameters and tensors based on extracted data.
        """
        (self.lb, self.ub, self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4,
         self.yAB, self.xBC, self.yCD, self.xDE, self.yEF, self.xFG, self.yGH, self.xHI, self.yIJ,
         self.xJK, self.yKL, self.xLA, self.xp, self.yp, self.up, self.vp) = self.train_data

        (self.D11, self.D21, self.D12, self.D22, self.D13, self.D23, self.D14, self.D24,
         self.Sigma121, self.Sigma122, self.Sigma123, self.Sigma124, self.Sigmaa11, self.Sigmaa21,
         self.Sigmaa12, self.Sigmaa22, self.Sigmaa13, self.Sigmaa23, self.Sigmaa14, self.Sigmaa24,
         self.vSigmaf21, self.vSigmaf22, self.vSigmaf23, self.vSigmaf24, self.Bz1, self.Bz2,
         self.layers, self.data_path, self.device) = self.param_data

        self.keff_list = []

        self.data_tensors = [self.convert_to_tensor(data, requires_grad=False)
                             for data in [self.D11, self.D21, self.D12, self.D22, self.D13,
                                          self.D23, self.D14, self.D24, self.Sigma121,
                                          self.Sigma122, self.Sigma123, self.Sigma124,
                                          self.Sigmaa11, self.Sigmaa21, self.Sigmaa12,
                                          self.Sigmaa22, self.Sigmaa13, self.Sigmaa23,
                                          self.Sigmaa14, self.Sigmaa24, self.vSigmaf21,
                                          self.vSigmaf22, self.vSigmaf23, self.vSigmaf24,
                                          self.Bz1, self.Bz2]]

        self.lb = self.convert_to_tensor(self.lb, requires_grad=False)
        self.ub = self.convert_to_tensor(self.ub, requires_grad=False)

        self.boundary_data = self.initialize_boundaries()
        self.prior_data = self.initialize_prior_data()

        self.lambda_1 = Variable(torch.tensor(0.1)).to(self.device)
        self.lambda_1.requires_grad_()

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.loss = None
        self.loss_total_list = []
        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None
        self.start_time = None

    def neural_network(self, x, y, weights, biases):
        """
        Define the architecture of the neural network.
        """
        num_layers = len(weights) + 1
        X = torch.cat((x, y), dim=1)
        X = self.shift_coordinates(X)

        for l in range(num_layers - 2):
            W, b = weights[l], biases[l]
            X = torch.tanh(torch.add(torch.matmul(X, W), b))

        W, b = weights[-1], biases[-1]
        Y = torch.add(torch.matmul(X, W), b)
        return Y

    def compute_uv(self, x, y):
        """
        Compute the neural network output (u, v) given the input (x, y).
        :param x: Input x
        :param y: Input y
        :return: u, v
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        uv = self.neural_network(x_tensor, y_tensor, self.weights, self.biases)
        return uv[:, 0], uv[:, 1]  # Assuming uv is a tensor of shape (N, 2), where N is the number of samples

    def shift_coordinates(self, coordinates):
        """
        Shift coordinates to the range [-1, 1] based on lower and upper bounds.
        """
        return 2.0 * (coordinates - self.lb) / (self.ub - self.lb) - 1.0

    def initialize_boundaries(self):
        """
        Initialize boundary points and return them as tensors.
        """
        boundary_points = {
            "AB": ((0 * self.yAB + 0, self.yAB),),
            "BC": ((self.xBC, 0 * self.xBC + 170),),
            "CD": ((0 * self.yCD + 70, self.yCD),),
            "DE": ((self.xDE, 0 * self.xDE + 150),),
            "EF": ((0 * self.yEF + 110, self.yEF),),
            "FG": ((self.xFG, 0 * self.xFG + 130),),
            "GH": ((0 * self.yGH + 130, self.yGH),),
            "HI": ((self.xHI, 0 * self.xHI + 110),),
            "IJ": ((0 * self.yIJ + 150, self.yIJ),),
            "JK": ((self.xJK, 0 * self.xJK + 70),),
            "KL": ((0 * self.yKL + 170, self.yKL),),
            "LA": ((self.xLA, 0 * self.xLA + 0),)
        }

        for key, values in boundary_points.items():
            x1, x2 = values[0]
            y1, y2 = values[0]
            x_concat = np.concatenate((x1, x2), 1)
            y_concat = np.concatenate((y1, y2), 1)

            setattr(self, f'x_{key}', self.load_data(x_concat[:, 0:1]))
            setattr(self, f'y_{key}', self.load_data(y_concat[:, 1:2]))

    def initialize_prior_data(self):
        """
        Initialize and return prior data as tensors.
        """
        X_p = np.concatenate((self.xp, self.yp), axis=1)
        x_p = self.convert_to_tensor(X_p[:, 0:1])
        y_p = self.convert_to_tensor(X_p[:, 1:2])
        u_p = self.convert_to_tensor(self.up)
        v_p = self.convert_to_tensor(self.vp)
        return x_p, y_p, u_p, v_p

    def convert_to_tensor(self, data, requires_grad=True):
        """
        Convert data to a PyTorch tensor and move it to the appropriate device.
        """
        tensor = torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)
        return tensor.to(self.device)

    def load_data(self, data):
        """
        Convert data to a PyTorch tensor and move it to the appropriate device.
        """
        tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        return tensor.to(self.device)

    def initialize_NN(self, layers):
        """
        Initialize neural network weights and biases.
        """
        weights, biases = [], []
        for i in range(len(layers) - 1):
            W = nn.init.xavier_normal_(torch.empty(layers[i], layers[i + 1])).to(self.device)
            b = torch.zeros(layers[i + 1]).to(self.device)
            weights.append(W.requires_grad_())
            biases.append(b.requires_grad_())
        return weights, biases

    # First-order derivative
    def compute_first_order_gradients(self, u, v, x, y):
        """
        Compute the first-order gradients of u and v with respect to x and y.
        :param u: Function u(x, y)
        :param v: Function v(x, y)
        :param x: Input tensor x
        :param y: Input tensor y
        :return: First-order gradients of u and v with respect to x and y
        """
        try:
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).clone().detach().to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True).clone().detach().to(self.device)

            # Compute gradients for u with respect to x and y
            u_x = autograd.grad(outputs=u.sum(), inputs=x_tensor, create_graph=True)[0]
            u_y = autograd.grad(outputs=u.sum(), inputs=y_tensor, create_graph=True)[0]

            # Compute gradients for v with respect to x and y
            v_x = autograd.grad(outputs=v.sum(), inputs=x_tensor, create_graph=True)[0]
            v_y = autograd.grad(outputs=v.sum(), inputs=y_tensor, create_graph=True)[0]

            # Check if gradients are None and replace with zeros
            if u_x is None or u_y is None or v_x is None or v_y is None:
                print("Error computing first-order gradients: Some gradients are None.")
                return None, None, None, None

            return u_x, u_y, v_x, v_y
        except Exception as e:
            print("Error computing first-order gradients:", e)
            return None, None, None, None

        # First-order and second-order derivatives

    def compute_second_order_gradients(self, u, v, x, y):
        u_x, u_y, v_x, v_y = self.compute_first_order_gradients(u, v, x, y)
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = autograd.grad(v_y.sum(), y, create_graph=True)[0]
        return u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy

    def out_net(self, x, y):
        """
        Compute the output u and v given inputs x and y.
        """
        u, v = self.compute_uv(x, y)
        return u, v

    def forward(self, x, y):
        u, v = self.out_net(x, y)
        return self.convert_to_numpy(u), self.convert_to_numpy(v)

    def convert_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy().squeeze()
    def compute_loss(self, predicted, alpha=1):
        """
        Compute the loss.
        :param predicted: Predicted tensor
        :param alpha: Weighting factor for the loss
        :return: Computed loss
        """
        if predicted is None:
            return torch.tensor(0.0, dtype=torch.float32).to(self.device)

        true = torch.zeros_like(predicted).to(self.device)
        loss = torch.sum(torch.pow(predicted - true, 2))
        return alpha * loss

    def calculate_loss_for_region(self, u, v, u_xx, u_yy, v_xx, v_yy, region_type=1, alpha=1):
        region_params = {
            1: (self.D11, self.D21, self.Sigmaa11, self.Sigma121, self.Bz1, self.Bz2, self.vSigmaf21),
            2: (self.D12, self.D22, self.Sigmaa12, self.Sigma122, self.Bz1, self.Bz2, self.vSigmaf22),
            3: (self.D13, self.D23, self.Sigmaa13, self.Sigma123, self.Bz1, self.Bz2, self.vSigmaf23),
            4: (self.D14, self.D24, self.Sigmaa14, self.Sigma124, self.Bz1, self.Bz2, self.vSigmaf24)
        }

        D1, D2, Sigmaa, Sigma, Bz1, Bz2, vSigmaf = region_params[region_type]

        f_u = -D1 * (u_xx + u_yy) + (Sigmaa + Sigma + D1 * Bz1) * u - (self.lambda_1 * vSigmaf * v)
        f_v = -D2 * (v_xx + v_yy) + (Sigmaa + D2 * Bz2) * v - (Sigma * u)

        loss = self.compute_loss(f_u, alpha=alpha) + self.compute_loss(f_v, alpha=alpha)
        return loss

    def compute_region_loss(self, x, y, region_type=1, alpha=1):
        u, v = self.out_net(x, y)
        u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy = self.compute_second_order_gradients(u, v, x, y)
        return self.calculate_loss_for_region(u, v, u_xx, u_yy, v_xx, v_yy, region_type, alpha)

    def compute_total_region_loss(self, alpha=1):
        losses = [
            self.compute_region_loss(self.x1, self.y1, region_type=1, alpha=alpha),
            self.compute_region_loss(self.x2, self.y2, region_type=2, alpha=alpha),
            self.compute_region_loss(self.x3, self.y3, region_type=3, alpha=alpha),
            self.compute_region_loss(self.x4, self.y4, region_type=4, alpha=alpha)
        ]
        return tuple(losses)

    def compute_prior_loss(self, x_p, y_p, u_p, v_p, alpha=1):
        u, v = self.out_net(x_p, y_p)
        loss = self.compute_loss(u, u_p, alpha=alpha) + self.compute_loss(v, v_p, alpha=alpha)
        return loss

    def compute_total_boundary_loss(self, alpha=1):
        """
        Compute the total boundary loss by summing the loss over all boundary segments.
        :param alpha: Weighting factor for the loss
        :return: Total boundary loss
        """
        boundary_points = [
            (self.xAB, self.yAB, 2),
            (self.xBC, self.yBC, 3),
            (self.xCD, self.yCD, 4),
            (self.xDE, self.yDE, 3),
            (self.xEF, self.yEF, 4),
            (self.xFG, self.yFG, 3),
            (self.xGH, self.yGH, 4),
            (self.xHI, self.yHI, 3),
            (self.xIJ, self.yIJ, 4),
            (self.xJK, self.yJK, 3),
            (self.xKL, self.yKL, 4),
            (self.xLA, self.yLA, 1)
        ]

        total_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        total_loss.requires_grad_()

        for x, y, boundary_type in boundary_points:
            u, v = self.out_net(x, y)
            u_x, u_y, v_x, v_y = self.compute_first_order_gradients(u, v, x, y)
            loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            loss.requires_grad_()
            if boundary_type == 1:
                loss = loss + self.compute_loss(u_y, alpha=alpha) + self.compute_loss(v_y, alpha=alpha)
            elif boundary_type == 2:
                loss = loss + self.compute_loss(u_x, alpha=alpha) + self.compute_loss(v_x, alpha=alpha)
            elif boundary_type == 3:
                D14_tensor = torch.tensor(self.D14, dtype=torch.float32).to(self.device)
                loss = loss + self.compute_loss(u_y + 0.4692 / D14_tensor * u, alpha=alpha) + self.compute_loss(
                    v_y + 0.4692 / D14_tensor * v, alpha=alpha)
            elif boundary_type == 4:
                D14_tensor = torch.tensor(self.D14, dtype=torch.float32).to(self.device)
                loss = loss + self.compute_loss(u_x + 0.4692 / D14_tensor * u, alpha=alpha) + self.compute_loss(
                    v_x + 0.4692 / D14_tensor * v, alpha=alpha)
            total_loss = total_loss + loss  # Using regular addition instead of in-place addition

        return total_loss

    def detach(self, data):
        return data.detach().cpu().numpy().squeeze()
    def calculate_losses(self, alpha_boundary, alpha_region, alpha_prior):
        # Calculate boundary loss
        loss_boundary = self.compute_total_boundary_loss(alpha=alpha_boundary)
        loss_boundary = self.detach(loss_boundary) / alpha_boundary

        # Calculate region losses
        loss1, loss2, loss3, loss4 = self.compute_total_region_loss(alpha=alpha_region)
        loss_regions = self.detach(loss1 + loss2 + loss3 + loss4) / alpha_region

        # Calculate prior loss
        loss_prior = self.loss_prior(self.x_p, self.y_p, self.u_p, self.v_p, alpha=alpha_prior)
        loss_prior = self.detach(loss_prior) / alpha_prior

        return loss_boundary, loss_regions, loss_prior

    def log_losses(self, loss_boundary, loss_regions, loss_prior, keff):
        # Log total loss and keff
        log_str1 = f"{self.optimizer_name} Iter: {self.nIter} Loss: {self.loss} Keff: {keff}"
        print(log_str1)

        # Log individual losses
        log_str2 = (f"Loss of boundary: {loss_boundary} "
                    f"Loss of regions: {loss_regions} "
                    f"Loss of prior: {loss_prior}")
        print(log_str2)

    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # Loss Weights
        alpha_boundary = 1
        alpha_region = 1
        alpha_prior = 10

        # Zero gradients
        self.optimizer.zero_grad()

        # Initialize total loss
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        # Calculate and accumulate losses
        loss_boundary, loss_regions, loss_prior = self.calculate_losses(alpha_boundary, alpha_region, alpha_prior)
        self.loss += loss_boundary + loss_regions + loss_prior

        # Backpropagation
        self.loss.backward()
        self.nIter += 1

        # Detach loss and append to loss_total_list
        loss = self.detach(self.loss)
        self.loss_total_list.append(loss)

        # Log loss and keff
        lambda_1_value = self.detach(self.lambda_1)
        keff = 1 / lambda_1_value
        self.log_losses(loss_boundary, loss_regions, loss_prior, keff)

        # Log iteration and time
        elapsed = time.time() - self.start_time
        print(f"Iter: {self.nIter}, Time: {elapsed:.4f}")
        self.start_time = time.time()

        # Plot loss and keff after 100 iterations
        if self.nIter == 100:
            self.plot_loss('../data')
            self.plot_keff('../data')

        return self.loss


    def set_optimizer(self, optimizer, nIter, scheduler, optimizer_name):
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.scheduler = scheduler
        diff_mean = []

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss

        if optimizer_name == 'LBFGS':
            for _ in range(nIter):
                self.optimizer.step(closure)
        elif optimizer_name == 'Adam':
            for it in range(nIter):
                self.optimize_one_epoch()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step(self.loss)

                # Check for convergence every 2000 iterations
                if len(self.loss_total_list) > 1 and (len(self.loss_total_list) - 1) % 2000 == 0:
                    diff = [abs(self.loss_total_list[i] - self.loss_total_list[i + 1]) for i in
                            range(len(self.loss_total_list) - 1)]
                    diff_mean.append(np.mean(diff[(2000 * ((len(self.loss_total_list) - 1) // 2000) - 2000): (
                                2000 * ((len(self.loss_total_list) - 1) // 2000))]))

                    # Check for convergence condition
                    if len(diff_mean) >= 2 and abs(diff_mean[-1] - diff_mean[-2]) < 0.1:
                        break



    def plot_loss(self, data_path):
        plt.figure()
        plt.plot(self.loss_total_list, label='Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Function Over Iterations')
        plt.legend()
        plt.savefig(data_path + '/total_loss.png', dpi=300)

    def plot_keff(self, data_path):
        print("Keff List:", self.keff_list)
        plt.figure()
        plt.plot(self.keff_list, label='k_eff')
        plt.xlabel('Iterations')
        plt.ylabel('k_eff')
        plt.title('k_eff Over Iterations')
        plt.legend()
        plt.savefig(data_path + '/k_eff.png', dpi=300)

        # Initialize training configuration and model
        train_config = NN_params(self.device, self.data_path)
        self.train_data, self.param_data = train_config.get_config_dicts()

        # Create and configure the Physics-Informed Neural Network model
        self.model = self
        self.model.to(self.device)


    def optimize_and_predict(self, adam_steps, lbfgs_steps):
        # Configure the optimizers
        adam_optimizer = torch.optim.Adam(
            params=self.weights + self.biases + [self.lambda_1],
            lr=1e-3
        )
        lbfgs_optimizer = torch.optim.LBFGS(
            params=self.weights + self.biases + [self.lambda_1],
            lr=1,
            max_iter=lbfgs_steps
        )

        # Train with Adam optimizer
        self.set_optimizer(adam_optimizer, adam_steps, None, "Adam")

        # Train with L-BFGS optimizer
        self.set_optimizer(lbfgs_optimizer, lbfgs_steps, None, "LBFGS")

        # Perform predictions
        x_pred, y_pred = self.load_prediction_data()
        u_pred, v_pred = self.generate_predictions(x_pred, y_pred)

        # Save predictions to MATLAB file
        self.save_predictions(u_pred, v_pred)

        # Plot and save loss and keff graphs
        self.plot_and_save_graphs()


    def load_prediction_data(self):
        # Load prediction data
        prediction_data = scipy.io.loadmat(self.data_path + '/2DIBP_X_Pred.mat')
        return prediction_data['x_pred'], prediction_data['y_pred']


    def generate_predictions(self, x_pred, y_pred):
        # Generate predictions using the trained model
        x, y = self.load_data(x), self.load_data(y)
        with torch.no_grad():
            u_pred, v_pred = self.forward(x, y)
        return u_pred, v_pred


    def save_predictions(self, u_pred, v_pred):
        # Save the predictions to a MATLAB file
        scipy.io.savemat(self.data_path + '/2DIBP_U_Pred.mat', {'u_pred': u_pred, 'v_pred': v_pred})


    def plot_and_save_graphs(self):
        # Plot and save the loss and keff graphs
        self.plot_loss(self.data_path)
        self.plot_keff(self.data_path)
