import os
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable
from NN_params import NN_params

matplotlib.use('Agg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PhysicsInformedNN(nn.Module):

    def __init__(self, train_dict, param_dict):
        super(PhysicsInformedNN, self).__init__()

        # Retrieve data
        (lb, ub, x1, y1, x2, y2, x3, y3, x4, y4, yAB, xBC, yCD, xDE, yEF, xFG,
         yGH, xHI, yIJ, xJK, yKL, xLA, xp, yp, up, vp) = tuple(train_dict.values())

        (D11, D21, D12, D22, D13, D23, D14,
         D24, Sigma121, Sigma122, Sigma123, Sigma124,
         Sigmaa11, Sigmaa21, Sigmaa12, Sigmaa22,
         Sigmaa13, Sigmaa23, Sigmaa14, Sigmaa24,
         vSigmaf21, vSigmaf22, vSigmaf23, vSigmaf24,
         Bz1, Bz2, self.layers, self.data_path, self.device) = tuple(param_dict.values())
        #load equation parameters
        self.load_equation_parameters(D11, D21, D12, D22, D13, D23, D14,
                                 D24, Sigma121, Sigma122, Sigma123, Sigma124,
                                 Sigmaa11, Sigmaa21, Sigmaa12, Sigmaa22,
                                 Sigmaa13, Sigmaa23, Sigmaa14, Sigmaa24,
                                 vSigmaf21, vSigmaf22, vSigmaf23, vSigmaf24,
                                 Bz1, Bz2)
        #load boundary coords
        self.load_boundary_coords(yAB, xBC, yCD, xDE, yEF, xFG,
         yGH, xHI, yIJ, xJK, yKL, xLA)

        # load prior data
        self.load_prior_data(xp, yp, up, vp)

        # upper and lower boundaries
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        coordinates = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # List of coordinate tuples

        # Loop through the list of coordinate tuples
        for i, (x, y) in enumerate(coordinates, start=1):
            # Set the attribute x_i with the result of data_loader applied to x
            setattr(self, f"x_{i}", self.data_loader(x))
            # Set the attribute y_i with the result of data_loader applied to y
            setattr(self, f"y_{i}", self.data_loader(y))



        # The initial value of keff is set to 0.1
        self.lambda_1 = Variable(torch.tensor(0.1)).to(self.device)
        self.lambda_1.requires_grad_()

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss = None
        self.loss_total_list = []
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

        self.start_time = None

    def load_prior_data(self,xp, yp, up, vp):
        # Prior data
        X_p = np.concatenate((xp, yp), 1)
        self.x_p = self.data_loader(X_p[:, 0:1])
        self.y_p = self.data_loader(X_p[:, 1:2])
        self.u_p = self.data_loader(up)
        self.v_p = self.data_loader(vp)

    #Equation parameters loading
    def load_equation_parameters(self,D11, D21, D12, D22, D13, D23, D14,
         D24, Sigma121, Sigma122, Sigma123, Sigma124,
         Sigmaa11, Sigmaa21, Sigmaa12, Sigmaa22,
         Sigmaa13, Sigmaa23, Sigmaa14, Sigmaa24,
         vSigmaf21, vSigmaf22, vSigmaf23, vSigmaf24,
         Bz1, Bz2):
        data_list = [
            ("D11", D11), ("D21", D21), ("D12", D12), ("D22", D22),
            ("D13", D13), ("D23", D23), ("D14", D14), ("D24", D24),
            ("Sigma121", Sigma121), ("Sigma122", Sigma122), ("Sigma123", Sigma123), ("Sigma124", Sigma124),
            ("Sigmaa11", Sigmaa11), ("Sigmaa21", Sigmaa21), ("Sigmaa12", Sigmaa12), ("Sigmaa22", Sigmaa22),
            ("Sigmaa13", Sigmaa13), ("Sigmaa23", Sigmaa23), ("Sigmaa14", Sigmaa14), ("Sigmaa24", Sigmaa24),
            ("vSigmaf21", vSigmaf21), ("vSigmaf22", vSigmaf22), ("vSigmaf23", vSigmaf23), ("vSigmaf24", vSigmaf24),
            ("Bz1", Bz1), ("Bz2", Bz2)
        ]
        for var_name, var_data in data_list:
            setattr(self, var_name, self.data_loader(var_data, requires_grad=False))

    #Boundary coordinates loading
    def load_boundary_coords(self,yAB, xBC, yCD, xDE, yEF, xFG,
         yGH, xHI, yIJ, xJK, yKL, xLA):
        boundary_data = {
            "AB": (0 * yAB + 0, yAB),
            "BC": (xBC, 0 * xBC + 170),
            "CD": (0 * yCD + 70, yCD),
            "DE": (xDE, 0 * xDE + 150),
            "EF": (0 * yEF + 110, yEF),
            "FG": (xFG, 0 * xFG + 130),
            "GH": (0 * yGH + 130, yGH),
            "HI": (xHI, 0 * xHI + 110),
            "IJ": (0 * yIJ + 150, yIJ),
            "JK": (xJK, 0 * xJK + 70),
            "KL": (0 * yKL + 170, yKL),
            "LA": (xLA, 0 * xLA + 0)
        }

        for label, (x_data, y_data) in boundary_data.items():
            setattr(self, f"x_{label}", self.data_loader(x_data[:, 0:1]))
            setattr(self, f"y_{label}", self.data_loader(y_data[:, 1:2]))

    # Initialize neural network
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = Variable(torch.zeros([1, layers[l + 1]],
                                     dtype=torch.float32)).to(self.device)
            b.requires_grad_()
            weights.append(W)
            biases.append(b)
        return weights, biases

    def detach(self, data):
        return data.detach().cpu().numpy().squeeze()

    def xavier_init(self, size):
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_()
        return W

    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float32)
        return x_tensor.to(self.device)


    def neural_net(self, x, y):
        # Concatenate x and y along the second dimension (dim=1)
        X = torch.cat((x, y), dim=1)

        # Apply coordinate shift to normalize the input data
        X = - 1 + 2.0 * (X - self.lb) / (self.ub - self.lb)

        # Iterate through all layers except the last one
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            # Apply linear transformation and activation function (tanh)
            X = torch.tanh(torch.matmul(X, W) + b)

        # For the last layer, only apply the linear transformation without activation
        final_W = self.weights[-1]
        final_b = self.biases[-1]
        Y = torch.matmul(X, final_W) + final_b

        return Y

    def NN_model(self, x, y):
        # Concatenate x and y along the second dimension (dim=1)
        X = torch.cat((x, y), dim=1)

        # Apply coordinate shift to normalize the input data
        X = -1 + 2.0 * (X - self.lb) / (self.ub - self.lb)

        # Iterate through all layers except the last one
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            # Apply linear transformation and activation function (tanh)
            X = torch.tanh(torch.matmul(X, W) + b)

        # For the last layer, only apply the linear transformation without activation
        final_W = self.weights[-1]
        final_b = self.biases[-1]
        uv = torch.matmul(X, final_W) + final_b

        # Split the output into u and v components
        u = uv[:, 0:1]  # Select the first column as u
        v = uv[:, 1:2]  # Select the second column as v

        return u, v

    # first-order derivative
    def compute_gradients(self, outputs, inputs):
        return autograd.grad(outputs.sum(), inputs, create_graph=True)[0]

    def first_order_derivative(self, u, v, x, y):
        u_x = self.compute_gradients(u, x)
        u_y = self.compute_gradients(u, y)
        v_x = self.compute_gradients(v, x)
        v_y = self.compute_gradients(v, y)
        return u_x, u_y, v_x, v_y

    def second_order_derivative(self, u, v, x, y):
        u_x, u_y, v_x, v_y = self.first_order_derivative(u, v, x, y)
        u_xx = self.compute_gradients(u_x, x)
        u_yy = self.compute_gradients(u_y, y)
        v_xx = self.compute_gradients(v_x, x)
        v_yy = self.compute_gradients(v_y, y)
        return u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy

    def forward(self, x, y):
        """
        Forward pass of the neural network.

        Args:
            x: Input tensor for the x-coordinate.
            y: Input tensor for the y-coordinate.

        Returns:
            Detached tensors for the u and v components of the output.
        """
        u, v = self.NN_model(x, y)  # Compute the u and v components of the output
        return u.detach(), v.detach()  # Detach u and v from the computation graph before returning

    def compute_equation_loss(self, u, v, u_xx, u_yy, v_xx, v_yy, type=1, alpha=1):
        # Define coefficients for different regions
        coefficients = {
            1: (self.D11, self.Sigmaa11, self.Sigma121, self.Bz1, self.vSigmaf21),
            2: (self.D12, self.Sigmaa12, self.Sigma122, self.Bz1, self.vSigmaf22),
            3: (self.D13, self.Sigmaa13, self.Sigma123, self.Bz1, self.vSigmaf23),
            4: (self.D14, self.Sigmaa14, self.Sigma124, self.Bz1, self.vSigmaf24)
        }

        # Get coefficients based on the type
        D, Sigmaa, Sigma12, Bz1, lambda_vSigmaf = coefficients[type]

        # Calculate f_u and f_v using the obtained coefficients
        f_u = -D * (u_xx + u_yy) + (Sigmaa + Sigma12 + D * Bz1) * u - (self.lambda_1 * lambda_vSigmaf * v)
        f_v = -self.D22 * (v_xx + v_yy) + (self.Sigmaa22 + self.D22 * self.Bz2) * v - (self.Sigma122 * u)

        # Compute loss
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        loss = loss + self.evaluate_loss(f_u, alpha=alpha)
        loss = loss + self.evaluate_loss(f_v, alpha=alpha)

        return loss


    def compute_regions_losses(self, alpha=1):
        loss_regions = []

        for region_type, (x, y) in zip(range(1, 5), [(self.x_1, self.y_1), (self.x_2, self.y_2),
                                                     (self.x_3, self.y_3), (self.x_4, self.y_4)]):
            # Compute second-order derivatives
            u, v = self.NN_model(x, y)
            u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy = self.second_order_derivative(u, v, x, y)

            # Compute loss for the region
            loss_region = self.compute_equation_loss(u, v, u_xx, u_yy, v_xx, v_yy, region_type, alpha=alpha)

            loss_regions.append(loss_region)

        return tuple(loss_regions)

    def compute_prior_loss(self, x_p, y_p, u_p, v_p, alpha=1):
        # Compute u and v predictions for given coordinates
        u, v = self.NN_model(x_p, y_p)

        # Compute loss for u and v predictions
        loss_u = self.evaluate_loss(u, u_p, alpha=alpha)
        loss_v = self.evaluate_loss(v, v_p, alpha=alpha)

        # Combine the losses
        loss = loss_u + loss_v

        return loss

    # Loss_boundary
    def compute_boundary_loss(self, alpha=1):
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()

        # Define a dictionary to map boundary types to loss computations
        loss_computations = {
            1: lambda u, v, u_y, v_y: (self.evaluate_loss(u_y, alpha=alpha), self.evaluate_loss(v_y, alpha=alpha)),
            2: lambda u, v, u_x, v_x: (self.evaluate_loss(u_x, alpha=alpha), self.evaluate_loss(v_x, alpha=alpha)),
            3: lambda u, v, u_y, v_y: (self.evaluate_loss(u_y + 0.4692 / self.D14 * u, alpha=alpha),
                                       self.evaluate_loss(v_y + 0.4692 / self.D24 * v, alpha=alpha)),
            4: lambda u, v, u_x, v_x: (self.evaluate_loss(u_x + 0.4692 / self.D14 * u, alpha=alpha),
                                       self.evaluate_loss(v_x + 0.4692 / self.D24 * v, alpha=alpha))
        }

        boundaries = [
            (self.x_AB, self.y_AB, 2), (self.x_BC, self.y_BC, 3),
            (self.x_CD, self.y_CD, 4), (self.x_DE, self.y_DE, 3),
            (self.x_EF, self.y_EF, 4), (self.x_FG, self.y_FG, 3),
            (self.x_GH, self.y_GH, 4), (self.x_HI, self.y_HI, 3),
            (self.x_IJ, self.y_IJ, 4), (self.x_JK, self.y_JK, 3),
            (self.x_KL, self.y_KL, 4), (self.x_LA, self.y_LA, 1)
        ]

        for x, y, type in boundaries:
            # Compute u and v predictions for given coordinates
            u, v = self.NN_model(x, y)
            u_x, u_y, v_x, v_y = self.first_order_derivative(u, v, x, y)

            # Compute loss for the boundary segment based on its type using the dictionary
            u_loss, v_loss = loss_computations[type](u, v, u_y, v_y) if type in [1, 3] else loss_computations[type](u,
                                                                                                                    v,
                                                                                                                    u_x,
                                                                                                                    v_x)

            # Add the computed losses to the total loss
            loss = loss + u_loss + v_loss

        return loss

    def evaluate_loss(self, pred_, true_=None, alpha=1):
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
        return alpha * self.loss_fn(pred_, true_)

    def calculate_losses(self, alpha_boundary, alpha_region, alpha_prior):
        # Calculate boundary loss
        loss_boundary = self.compute_boundary_loss(alpha=alpha_boundary)
        loss_boundary = self.detach(loss_boundary) / alpha_boundary

        # Calculate region losses
        loss1, loss2, loss3, loss4 = self.compute_regions_losses(alpha=alpha_region)
        loss_regions = self.detach(loss1 + loss2 + loss3 + loss4) / alpha_region

        # Calculate prior loss
        loss_prior = self.compute_prior_loss(self.x_p, self.y_p, self.u_p, self.v_p, alpha=alpha_prior)
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

    def train(self):
        if self.start_time is None:
            self.start_time = time.time()

        # Loss Weights
        alpha_boundary = 1
        alpha_region = 1
        alpha_prior = 10

        # Zero gradients
        self.optimizer.zero_grad()

        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()
        # Calculate and accumulate losses

        cum_loss = self.loss
        loss_boundary, loss_regions, loss_prior = self.calculate_losses(alpha_boundary, alpha_region, alpha_prior)
        self.loss = cum_loss + loss_boundary + loss_regions + loss_prior

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
                self.train()
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
        """
        Generate predictions using the trained model.

        Args:
            x_pred: Input tensor for the x-coordinate.
            y_pred: Input tensor for the y-coordinate.

        Returns:
            Detached tensors for the u and v components of the output.
        """
        x, y = self.load_data(x_pred), self.load_data(y_pred)
        with torch.no_grad():
            # Compute the u and v components of the output using NN_model
            u, v = self.NN_model(x, y)
            # Detach u and v from the computation graph before returning
            return u.detach(), v.detach()

    def save_predictions(self, u_pred, v_pred):
        # Save the predictions to a MATLAB file
        scipy.io.savemat(self.data_path + '/U_Pred_Data.mat', {'u_pred': u_pred, 'v_pred': v_pred})

    def plot_and_save_graphs(self):
        # Plot and save the loss and keff graphs
        self.plot_loss(self.data_path)
        self.plot_keff(self.data_path)


