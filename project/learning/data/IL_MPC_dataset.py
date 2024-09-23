import torch
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import os
import pickle
from torch.utils.data import Dataset

class MPCTrajectoryDataset(Dataset):
    
    NORMALIZATION_FILE = "normalization_stats.pkl"
    
    def __init__(self,
                 data_file: str,
                 use_set_points: bool,
                 contact_conditioned: bool,
                 normalize: bool = False,
                 normalization_file: str = None):
        """
        PyTorch Dataset for loading MPC trajectory data and converting world-frame 
        quantities to the robot's base frame, with optional normalization.

        Args:
            data_file (str): Path to the saved `.npz` file containing the trajectory data.
            use_set_points (bool): Whether to use set points as targets.
            contact_conditioned (bool): Whether the input is conditioned on contacts.
            normalize (bool): Whether to normalize the data. Defaults to False.
            normalization_file (str): Path to save or load normalization parameters (mean and std).
        """
        self.use_set_points = use_set_points
        self.contact_conditioned = contact_conditioned
        self.normalize = normalize
        if normalization_file is None:
            data_dir, _ = os.path.split(data_file)
            self.normalization_file = os.path.join(data_dir, MPCTrajectoryDataset.NORMALIZATION_FILE)
        else:
            self.normalization_file = normalization_file
        
         # Load the data from the file
        data = np.load(data_file)
        self.q_data = torch.tensor(data['q'], dtype=torch.float32)
        self.v_data = torch.tensor(data['v'], dtype=torch.float32)
        self.feet_pos_w = torch.tensor(data['feet_pos_w'], dtype=torch.float32)
        self.target_contact_w = torch.tensor(data['target_cnt_w'], dtype=torch.float32)
        self.time_to_contact = torch.tensor(data['time_to_cnt'], dtype=torch.float32)
        self.feet_contact = torch.tensor(data['foot_cnt'], dtype=torch.float32)
        self.v_des = torch.tensor(data['v_des'], dtype=torch.float32)
        self.w_des = torch.tensor(data['w_des'], dtype=torch.float32)
        self.gait_phase = torch.tensor(data['gait_phase'], dtype=torch.float32)
        self.tau = torch.tensor(data['tau'], dtype=torch.float32)
        self.tau_set_points = torch.tensor(data['pd_set_points'], dtype=torch.float32)

        # Initialize empty tensors for base-frame transformations
        N = len(self.q_data)
        self.feet_pos_b = torch.empty((N, 12), dtype=torch.float32)
        self.target_contact_b = torch.empty((N, 12), dtype=torch.float32)

        # Transform all data from world frame to base frame
        for i in range(N):
            q = self.q_data[i].numpy()
            target_w = self.target_contact_w[i].numpy().reshape(-1, 3)
            feet_w = self.feet_pos_w[i].numpy().reshape(-1, 3)

            # Perform transformation and store in the base frame tensors
            self.target_contact_b[i, :] = torch.tensor(self.transform_to_base_frame(q, target_w), dtype=torch.float32).reshape(-1)
            self.feet_pos_b[i, :] = torch.tensor(self.transform_to_base_frame(q, feet_w), dtype=torch.float32).reshape(-1)
        
        # Remove absolute position
        self.q_data = self.q_data[:, 3:]

        # Normalize the data if requested
        if self.normalize:
            if self.normalization_file and os.path.exists(self.normalization_file):
                # Load the normalization parameters if they exist
                self.mean_std = self.load_normalization(self.normalization_file)
            else:
                # Compute normalization parameters and save them
                self.mean_std = self.compute_normalization()
                if self.normalization_file:
                    self.save_normalization(self.normalization_file, self.mean_std)
            
            # Apply normalization to the input data
            self.apply_normalization()

    def __len__(self):
        return len(self.q_data)

    def compute_normalization(self):
        """
        Compute the mean and standard deviation of the data for normalization.

        Returns:
            dict: Dictionary containing the mean and standard deviation for each feature.
        """
        inputs = []
        for i in range(len(self.q_data)):
            input_data = torch.cat((
                self.q_data[i],
                self.v_data[i],
                self.feet_pos_b[i],
                self.target_contact_b[i],
                self.feet_contact[i],
                self.time_to_contact[i],
            )).reshape(-1)
            inputs.append(input_data)

        inputs = torch.stack(inputs)
        mean = inputs.mean(dim=0)
        std = inputs.std(dim=0)

        return {"mean": mean, "std": std}

    def save_normalization(self, path: str, mean_std: dict):
        """
        Save the normalization parameters (mean and std) to a file.

        Args:
            path (str): Path to save the normalization parameters.
            mean_std (dict): Dictionary containing the mean and std tensors.
        """
        with open(path, 'wb') as f:
            pickle.dump(mean_std, f)

    def load_normalization(self, path: str) -> dict:
        """
        Load normalization parameters (mean and std) from a file.

        Args:
            path (str): Path to load the normalization parameters from.

        Returns:
            dict: Dictionary containing the mean and std tensors.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def apply_normalization(self):
        """
        Normalize the input data using the precomputed mean and std.
        """
        mean = self.mean_std["mean"]
        std = self.mean_std["std"]
        
        inputs = []
        for i in range(len(self.q_data)):
            input_data = torch.cat((
                self.q_data[i],
                self.v_data[i],
                self.feet_pos_b[i],
                self.target_contact_b[i],
                self.feet_contact[i],
                self.time_to_contact[i],
            )).reshape(-1)
            inputs.append(input_data)

        inputs = torch.stack(inputs)
            
        # Normalize in place
        normalized_input = (inputs - mean) / (std + 1e-8)  # Add epsilon for numerical stability
        
        sections = [
        len(self.q_data[i].reshape(-1)),
        len(self.v_data[i].reshape(-1)),
        len(self.feet_pos_b[i].reshape(-1)),
        len(self.target_contact_b[i].reshape(-1)),
        len(self.feet_contact[i].reshape(-1)),
        len(self.time_to_contact[i].reshape(-1))
        ]
        (self.q_data,
        self.v_data,
        self.feet_pos_b,
        self.target_contact_b,
        self.feet_contact,
        self.time_to_contact) = torch.split(normalized_input, sections, dim=-1)
        
    @staticmethod
    def transform_to_base_frame(q, points_w):
        """
        Transform points from world frame to the robot's base frame.

        Args:
            q (np.ndarray): Robot state [x, y, z, qx, qy, qz, qw].
            points_w (np.ndarray): Points in world frame, shape [N, 3].

        Returns:
            np.ndarray: Points in the base frame, shape [N, 3].
        """
        # Compute the transformation matrix from world to base frame
        B_T_W = pin.XYZQUATToSE3(q[:7]).inverse()

        # Homogeneous transformation
        ones = np.ones((points_w.shape[0], 1))
        points_w_homogeneous = np.hstack((points_w, ones))
        points_b_homogeneous = B_T_W @ points_w_homogeneous.T
        points_b = points_b_homogeneous[:3, :].T
        return points_b
    
    def plot_histogram(self, variable_name: str):
        """
        Plot the histograms of each dimension for the specified variable from the dataset.

        Args:
            variable_name (str): The name of the variable to plot. 
                                It should match one of the dataset's attributes.
        """
        # Ensure the variable exists in the dataset
        if not hasattr(self, variable_name):
            raise ValueError(f"Variable '{variable_name}' not found in the dataset.")
        
        # Extract the variable's data
        variable_data = getattr(self, variable_name)
        if isinstance(variable_data, torch.Tensor):
            variable_data = variable_data.numpy().reshape(len(variable_data), -1)

        # Get the number of dimensions in the variable
        num_dimensions = variable_data.shape[-1] if len(variable_data.shape) > 1 else 1

        # Create subplots
        cols = 3  # Number of columns for subplots
        rows = (num_dimensions + cols - 1) // cols  # Calculate number of rows needed
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if num_dimensions > 1 else [axes]

        # Plot the histograms for each dimension
        for i in range(num_dimensions):
            ax = axes[i]
            ax.hist(variable_data[:, i].flatten(), bins=50, color='blue', alpha=0.7)
            ax.set_title(f"Histogram of {variable_name} (Dimension {i+1})")
            ax.set_xlabel(variable_name)
            ax.set_ylabel("Frequency")

        # Remove any unused subplots
        for i in range(num_dimensions, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
    
    def __getitem__(self, idx):
        """
        Get the data for a single timestep.

        Args:
            idx (int): Index for the timestep.

        Returns:
            dict: Dictionary containing concatenated tensors of data.
        """
        # Inputs
        if self.contact_conditioned:
            inputs = torch.cat((
                self.q_data[idx],
                self.v_data[idx],
                self.feet_pos_b[idx],
                self.target_contact_b[idx],
                self.feet_contact[idx],
                self.time_to_contact[idx],
            ))
        else:  # Velocity conditioned
            inputs = torch.cat((
                self.q_data[idx],
                self.v_data[idx],
                self.feet_pos_b[idx],
                self.v_des[idx],
                self.w_des[idx],
                self.gait_phase[idx],
            ))
        
        # Targets
        if self.use_set_points:
            targets = self.tau_set_points[idx]
        else:
            targets = self.tau[idx]
        
        return {
            "input": inputs,
            "target": targets,
        }
