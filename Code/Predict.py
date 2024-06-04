import scipy.io
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Define data path
data_path = '../data'

# Load data
data_true = scipy.io.loadmat(f"{data_path}/Prior_Data.mat")
data_pred = scipy.io.loadmat(f"{data_path}/U_Pred_Data.mat")


# Function to reshape and convert numpy array to list
def nptolist(arr):
    return arr.flatten().tolist()


# load prior data and convert to lists
x_true = data_true['x'].flatten().tolist()
y_true = data_true['y'].flatten().tolist()
u_true = data_true['phi1'].flatten().tolist()
v_true = data_true['phi2'].flatten().tolist()

# Create prior dictionary
X_true = list(zip(x_true, y_true))
dictu = dict(zip(X_true, u_true))
dictv = dict(zip(X_true, v_true))


# Function to generate mesh grid and convert to list
def generate_meshgrid(start, end, step):
    values = np.arange(start, end, step)
    grid_x, grid_y = np.meshgrid(values, values)
    return grid_x, grid_y, np.hstack((grid_x.flatten()[:, None], grid_y.flatten()[:, None]))


# Generate mesh grid for X_star and Y_star
X_star, Y_star, XY_star = generate_meshgrid(0.0, 171.0, 1.0)

# Convert mesh grid to lists
Xstar_list = X_star.flatten().tolist()
Ystar_list = Y_star.flatten().tolist()


# Function to search prior dictionary and assign u_prior and v_prior values
def search_prior_dictionary(Xstar_list, Ystar_list, dictu, dictv):
    u_prior = []
    v_prior = []

    for x, y in zip(Xstar_list, Ystar_list):
        # Swap values if x <= y
        if x <= y:
            x, y = y, x

        # Check conditions and assign values accordingly
        if (151 <= x and y >= 71) or (131 <= x <= 150 and y >= 111) or (111 <= x <= 130 and y >= 131) or (
                71 <= x <= 110 and y >= 151):
            u_prior.append(0)
            v_prior.append(0)
        else:
            u_prior.append(dictu.get((x, y), 0))
            v_prior.append(dictv.get((x, y), 0))

    return u_prior, v_prior


# Search prior dictionary and assign u_prior and v_prior values
u_prior, v_prior = search_prior_dictionary(Xstar_list, Ystar_list, dictu, dictv)


# Function to reshape and flatten prediction arrays
def reshape_and_flatten(arr):
    return arr.reshape(-1).tolist()


# Get predicted values and reshape them
u_pred = reshape_and_flatten(data_pred['u_pred'])
v_pred = reshape_and_flatten(data_pred['v_pred'])

# Combine Xstar_list and Ystar_list into a list of tuples for XY_pred
XY_pred = list(zip(Xstar_list, Ystar_list))

# Create predicted dictionaries
dictu_pred = {xy: u for xy, u in zip(XY_pred, u_pred)}
dictv_pred = {xy: v for xy, v in zip(XY_pred, v_pred)}


def search_predicted_dictionary(Xstar_list, Ystar_list, dictu_pred, dictv_pred):
    u_prediction = []
    v_prediction = []

    for x, y in zip(Xstar_list, Ystar_list):
        if x <= y:
            x, y = y, x

        if (x >= 151 and y >= 71) or (131 <= x <= 150 and y >= 111) or (111 <= x <= 130 and y >= 131) or (
                71 <= x <= 110 and y >= 151):
            u_prediction.append(0)
            v_prediction.append(0)
        else:
            u_prediction.append(dictu_pred[(x, y)])
            v_prediction.append(dictv_pred[(x, y)])

    return np.array(u_prediction), np.array(v_prediction)


def create_meshgrid(start, end, step):
    axis = np.arange(start, end, step).reshape(-1, 1)
    return np.meshgrid(axis, axis)


def interpolate_grid(XY, values, X_star, Y_star):
    values = np.array(values)
    return griddata(XY, values.flatten(), (X_star, Y_star), method='cubic')


def plot_heatmap(data, title):
    plt.imshow(data, extent=(0, 170, 0, 170), origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()


# Example usage:
# Assuming XY_star, Xstar_list, Ystar_list, dictu_pred, dictv_pred, u_prior, v_prior are defined elsewhere

# Convert u_prior and v_prior to numpy arrays
u_prior = np.array(u_prior)
v_prior = np.array(v_prior)

# Search predicted dictionary
u_prediction, v_prediction = search_predicted_dictionary(Xstar_list, Ystar_list, dictu_pred, dictv_pred)

# Create meshgrid
X_star, Y_star = create_meshgrid(0.0, 171.0, 1.0)

# Interpolate grid
U_star = interpolate_grid(XY_star, u_prior, X_star, Y_star)
U_pred = interpolate_grid(XY_star, u_prediction, X_star, Y_star)
V_star = interpolate_grid(XY_star, v_prior, X_star, Y_star)
V_pred = interpolate_grid(XY_star, v_prediction, X_star, Y_star)

# Plot heatmap
plot_heatmap(U_star, 'U Star')
plot_heatmap(U_pred, 'U Prediction')
plot_heatmap(V_star, 'V Star')
plot_heatmap(V_pred, 'V Prediction')


def plot_and_save_contourf(X_star, Y_star, data, filename):
    cset = plt.contourf(X_star, Y_star, data)
    plt.colorbar(cset)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(filename, dpi=300)
    plt.show()


# Plot and save U_star
plot_and_save_contourf(X_star, Y_star, U_star, data_path + '/U_star.png')

# Plot and save U_pred
plot_and_save_contourf(X_star, Y_star, U_pred, data_path + '/U_pred.png')

# Plot and save absolute error of U
plot_and_save_contourf(X_star, Y_star, np.abs(U_star - U_pred), data_path + '/U_error.png')

# Plot and save V_star
plot_and_save_contourf(X_star, Y_star, V_star, data_path + '/V_star.png')

# Plot and save V_pred
plot_and_save_contourf(X_star, Y_star, V_pred, data_path + '/V_pred.png')

# Plot and save absolute error of V
plot_and_save_contourf(X_star, Y_star, np.abs(V_star - V_pred), data_path + '/V_error.png')

# Compute relative error for U
mask_u = np.abs(U_star) > 0  # Create a mask to avoid division by zero
relative_error_u = np.zeros_like(U_star)
relative_error_u[mask_u] = np.abs(U_star[mask_u] - U_pred[mask_u]) / np.abs(U_star[mask_u])

# Compute relative error for V
mask_v = np.abs(V_star) > 0  # Create a mask to avoid division by zero
relative_error_v = np.zeros_like(V_star)
relative_error_v[mask_v] = np.abs(V_star[mask_v] - V_pred[mask_v]) / np.abs(V_star[mask_v])

# Compute L2 norms of relative error
RE_L2_u = np.linalg.norm(relative_error_u, 2) / np.sqrt(np.sum(mask_u))
RE_L2_v = np.linalg.norm(relative_error_v, 2) / np.sqrt(np.sum(mask_v))

print('RE_L2 for U: %e' % (RE_L2_u))
print('RE_L2 for V: %e' % (RE_L2_v))

# Compute maximum absolute error for U and V components
max_abs_error_u = np.max(np.abs(U_star - U_pred))
max_abs_error_v = np.max(np.abs(V_star - V_pred))

# Compute RE_Linf for U and V components
RE_Linf_u = max_abs_error_u / np.max(np.abs(U_star))
RE_Linf_v = max_abs_error_v / np.max(np.abs(V_star))

print('RE_Linf for U: %e' % (RE_Linf_u))
print('RE_Linf for V: %e' % (RE_Linf_v))
