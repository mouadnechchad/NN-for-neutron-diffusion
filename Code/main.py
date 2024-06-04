import torch
import time
from build_model import PhysicsInformedNN
from NN_params import NN_params

def main():
    # Use CUDA if available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device Activated:', device)

    start_time = time.time()

    # Create and configure the PINN model
    train_config = NN_params(device, '../data')
    train_dict, param_dict = train_config.get_config_data()

    print("Device just before initializing PhysicsInformedNN:", device)  # Add this line
    model = PhysicsInformedNN(train_dict, param_dict)

    model.optimize_and_predict(40000, 20000)  # Adjust steps as needed

    elapsed = time.time() - start_time
    print('Training time: %.4f seconds' % elapsed)

if __name__ == "__main__":
    print("Physics-Informed Neural Network (PINN) Training and Prediction")
    print("--------------------------------------------------------------")
    print("This script trains a PINN model using the provided data and")
    print("performs predictions. It uses both Adam and L-BFGS optimizers")
    print("to optimize the model's parameters.")
    print("--------------------------------------------------------------")
    main()
