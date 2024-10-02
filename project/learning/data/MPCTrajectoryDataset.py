from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import os
import pickle
from torch.utils.data import Dataset

class MPCTrajectoryDataset(Dataset):
    
    NORMALIZATION_FILE = "normalization_stats.pkl"    
    DATA_FILE = "data.npz"
    STATE_Q = "q"
    STATE_V = "v"
    FEET_POS = "feet_pos_w"
    FEET_CONTACT = "foot_cnt"
    GAIT_PHASE = "gait_phase"
    EXTERNAL_FORCES = "f_ext"
    TARGET_CONTACT = "target_cnt_w"
    TIME_TO_CONTACT = "time_to_cnt"
    TAU = "tau"
    PD_SET_POINTS = "pd_set_points"
    V_DES = "v_des"
    W_DES = "w_des"
    TIME = "time"

    DEFAULT_STATE_VARIABLES = [
        "gravity_b",
        "qj",
        "v",
        "feet_pos_b",
    ]
    
    VARIABLE_TO_NORMALIZE = [
        "gravity_b",
        "qj",
        "v",
        "feet_pos_b",
        "tau",
        "pd_set_points",
        "target_contact_b",
        "time_to_contact",
        "f_ext",
    ]
    
    def __init__(self,
                 data_dir: str,
                 use_set_points: bool,
                 contact_conditioned: bool,
                 state_variables: List[str] = [],
                 history_variables: List[str] = [],
                 normalize: bool = False,
                 sigma_noise: float = 0., 
                 history_length: int = 0,
                 CVAE: bool = False,
                 **kwargs,
                 ):  # New argument for history length
        """
        PyTorch Dataset for loading MPC trajectory data and converting world-frame 
        quantities to the robot's base frame, with optional normalization.

        Args:
            data_file (str): Path to the saved `.npz` file containing the trajectory data.
            use_set_points (bool): Whether to use set points as targets.
            contact_conditioned (bool): Whether the input is conditioned on contacts.
            normalize (bool): Whether to normalize the data. Defaults to False.
            sigma_noise (float): Add noise to the inputs in the getitem.
            history_length (int): Number of previous timesteps to include in the input state history.
            CVAE (bool): Return sample for CVAE trainer.
            normalization_file (str): Path to save or load normalization parameters (mean and std).
        """
        self.use_set_points = use_set_points
        self.contact_conditioned = contact_conditioned
        self.normalize = normalize
        self.sigma_noise = sigma_noise
        self.history_length = history_length
        self.CVAE = CVAE
               
        # Load the data from the file
        data_file = os.path.join(data_dir, MPCTrajectoryDataset.DATA_FILE)
        data = np.load(data_file)
        self.time = torch.tensor(data['time'], dtype=torch.float32)
        self.q, self.qj = torch.tensor(data['q'], dtype=torch.float32).split([7, 12], dim=-1)
        self.v = torch.tensor(data['v'], dtype=torch.float32)
        self.feet_pos_w = torch.tensor(data['feet_pos_w'], dtype=torch.float32)
        self.target_contact_w = torch.tensor(data['target_cnt_w'], dtype=torch.float32)
        self.time_to_contact = torch.tensor(data['time_to_cnt'], dtype=torch.float32)
        self.feet_contact = torch.tensor(data['foot_cnt'], dtype=torch.float32)
        self.v_des = torch.tensor(data['v_des'], dtype=torch.float32)
        self.w_des = torch.tensor(data['w_des'], dtype=torch.float32)
        self.gait_phase = torch.tensor(data['gait_phase'], dtype=torch.float32)
        self.tau = torch.tensor(data['tau'], dtype=torch.float32)
        self.pd_set_points = torch.tensor(data['pd_set_points'], dtype=torch.float32)
        self.f_ext = torch.tensor(data['f_ext'], dtype=torch.float32)
        self.N = len(self.q)
        self._clean_nan()
        
        # Initialize empty tensors for base-frame transformations
        self.gravity_b = torch.empty((self.N, 3), dtype=torch.float32)
        self.feet_pos_b = torch.empty((self.N, 12), dtype=torch.float32)
        self.target_contact_b = torch.empty((self.N, 12), dtype=torch.float32)

        # Transform all data from world frame to base frame
        self._batch_world_to_base_frame()
        
        # Precompute history indices
        if self.history_length > 0:
            self.history_indices = self._precompute_history_indices()
        
        # Create state tensor
        # Remove absolute position and orientation
        if not state_variables:
            state_variables = MPCTrajectoryDataset.DEFAULT_STATE_VARIABLES
        if not history_variables:
            history_variables = MPCTrajectoryDataset.DEFAULT_STATE_VARIABLES
            
        # Normalize the data if requested
        if self.normalize:
            data_dir, _ = os.path.split(data_dir)
            self.normalization_file_path = os.path.join(data_dir, MPCTrajectoryDataset.NORMALIZATION_FILE)
            # Compute normalization parameters and save them
            self.mean_std = self.compute_normalization()
            
            if self.normalization_file_path:
                self.save_normalization(self.normalization_file_path, self.mean_std)
    
            # Apply normalization to the input data
            self.apply_normalization()
            
        self.state = self._concatenate_tensor_from_variable_names(state_variables)
        self.state_history = self._concatenate_tensor_from_variable_names(history_variables)
        
    def _clean_nan(self):
        # Create boolean masks where tau and tau_set_points are not NaN
        mask_no_nan_tau = ~np.isnan(self.tau).any(axis=1).bool()  # Check for NaNs in tau
        mask_no_nan_tau_set_points = ~np.isnan(self.pd_set_points).any(axis=1).bool()  # Check for NaNs in tau_set_points

        # Combine the masks using logical AND to keep only rows where both are valid
        mask_no_nan = mask_no_nan_tau & mask_no_nan_tau_set_points

        # Apply the mask to filter out rows with NaNs
        self.q = self.q[mask_no_nan]
        self.v = self.v[mask_no_nan]
        self.feet_pos_w = self.feet_pos_w[mask_no_nan]
        self.target_contact_w = self.target_contact_w[mask_no_nan]
        self.time_to_contact = self.time_to_contact[mask_no_nan]
        self.feet_contact = self.feet_contact[mask_no_nan]
        self.v_des = self.v_des[mask_no_nan]
        self.w_des = self.w_des[mask_no_nan]
        self.gait_phase = self.gait_phase[mask_no_nan]
        self.tau = self.tau[mask_no_nan]
        self.pd_set_points = self.pd_set_points[mask_no_nan]
        
    def _concatenate_tensor_from_variable_names(self, variable_names: List[str]) -> torch.Tensor:
        """
        Dynamically compute a tensor based on the provided variable names using exec().
        
        Args:
            variable_names (List[str]): List of variable names to include in the tensor.
        
        Returns:
            torch.Tensor: Concatenated tensor based on the provided variable names.
        """
        components = []
        for var_name in variable_names:
            if var_name in self.__dict__:
                # Dynamically access the variable by its name
                exec(f"components.append(self.{var_name}.reshape(self.N, -1))")
            else:
                raise ValueError(f"Unknown variable: {var_name}")
        
        # Concatenate all the selected variables into a single tensor
        return torch.cat(components, dim=-1)

    def compute_normalization(self):
        """
        Compute the mean and standard deviation of the data for normalization.

        Returns:
            dict: Dictionary containing the mean and standard deviation for each feature.
        """
        mean_stats = {}
        std_stats = {}
        for var_name in MPCTrajectoryDataset.VARIABLE_TO_NORMALIZE:
            if var_name in self.__dict__:
                # Dynamically access the variable by its name
                exec(f"mean_stats['{var_name}'] = self.{var_name}.reshape(self.N, -1).mean(dim=0)")
                exec(f"std_stats['{var_name}'] = self.{var_name}.reshape(self.N, -1).std(dim=0)")
            else:
                raise ValueError(f"Unknown variable: {var_name}")
            
        normalization_stats = {"mean" : mean_stats, "std" : std_stats}
        return normalization_stats

    def apply_normalization(self):
        """
        Normalize the input data using the precomputed mean and std.
        """
        mean_dict = self.mean_std["mean"]
        std_dict = self.mean_std["std"]
        
        for var_name in MPCTrajectoryDataset.VARIABLE_TO_NORMALIZE:
            if var_name in self.__dict__:
                # Dynamically access the variable by its name
                mean, std = mean_dict[var_name], std_dict[var_name]
                exec(f"self.{var_name} = self.{var_name}.reshape(self.N, -1)")
                exec(f"self.{var_name} = (self.{var_name} - mean) / (std + 1.0e-8)")
            else:
                raise ValueError(f"Unknown variable: {var_name}")

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

    @staticmethod
    def batch_transform_to_base_frame(q_batch, points_w_batch):
        """
        Batch transform points from world frame to the robot's base frame for multiple states.

        Args:
            q_batch (np.ndarray): Batch of robot states, shape [N, 7] where each row is [x, y, z, qx, qy, qz, qw].
            points_w_batch (np.ndarray): Batch of points in world frame, shape [N, M, 3] where M is the number of points.

        Returns:
            np.ndarray: Batch of points transformed to base frame, shape [N, M, 3].
        """
        # Compute the inverse transformation matrices from world to base frame for each q in the batch
        B_T_W_batch = np.array([pin.XYZQUATToSE3(q[:7]).inverse().homogeneous for q in q_batch])
        
        # Homogeneous transformation: Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w_batch.shape[0], points_w_batch.shape[1], 1))
        points_w_homogeneous = np.concatenate((points_w_batch, ones), axis=-1)  # Shape [N, M, 4]

        # Apply the batch transformation: B_T_W_batch @ points_w_homogeneous
        points_b_homogeneous = np.einsum('nij,nmj->nmi', B_T_W_batch, points_w_homogeneous)
        
        # Convert back to 3D coordinates by dropping the homogeneous coordinate
        points_b = points_b_homogeneous[:, :, :3]
        
        return points_b

    def _batch_world_to_base_frame(self):
        """
        Process all data from world frame to base frame using batch matrix operations.
        """
        # Prepare gravity points in world frame for all states
        gravity_w_batch = np.array([np.array([[0., 0., -1.]]) + q[:3] for q in self.q.numpy()])

        # Concatenate all points in world frame
        points_w_batch = np.concatenate((
            self.target_contact_w.numpy().reshape(-1, 4, 3),  # Target contact points
            self.feet_pos_w.numpy().reshape(-1, 4, 3),        # Feet positions
            gravity_w_batch                                   # Gravity points
        ), axis=1)  # Shape [N, 9, 3] (4 target contact points, 4 feet positions, 1 gravity vector)

        # Apply batch transformation
        points_b_batch = self.batch_transform_to_base_frame(self.q.numpy(), points_w_batch)

        # Split the transformed points into their respective tensors
        self.target_contact_b = torch.tensor(points_b_batch[:, :4].reshape(-1, 12), dtype=torch.float32)
        self.feet_pos_b = torch.tensor(points_b_batch[:, 4:8].reshape(-1, 12), dtype=torch.float32)
        self.gravity_b = torch.tensor(points_b_batch[:, 8].reshape(-1, 3), dtype=torch.float32)
            
    def _precompute_history_indices(self):
        """
        Precompute history indices for each timestep and store them in a tensor.
        The history indices for each timestep point to the valid states to include in the history.
        """
        N = len(self.time)  # Number of timesteps
        device = self.time.device
        
        # Create base indices where each row corresponds to the current index minus [1, ..., history_length]
        base_indices = torch.arange(1, self.history_length + 1, device=device).unsqueeze(0).expand(N, -1)
        indices = torch.arange(N, device=device).unsqueeze(1) - base_indices
        
        # Clamp indices to ensure they don't go below 0
        indices = torch.clamp(indices, min=0)

        # Create a mask for valid indices (to handle episode boundaries)
        # Compare the time of the current index with the time of previous indices
        time_expanded = self.time.expand(N, self.history_length)  # [N, history_length]
        time_indices = self.time[indices].squeeze(-1)  # Get times at the precomputed indices [N, history_length]

        mask = time_indices > time_expanded  # True if the history belongs to the same or earlier time
        
        # Get the last valid index for each timestep
        argmax = torch.argmax(mask.int(), dim=-1, keepdim=True)
        last_valid_id = torch.gather(indices, 1, argmax)  # Gather the last valid index for each row
        last_valid_id = torch.clamp(last_valid_id + 1, max=N)  # Gather the last valid index for each row

        # Broadcast last_valid_id to the same shape as indices to replace invalid entries
        last_valid_id_broadcast = last_valid_id.expand(-1, self.history_length)

        # Replace invalid entries with the last valid index
        indices[mask] = last_valid_id_broadcast[mask]

        return indices

    def _get_history(self, idx):
        """
        Retrieve the state history up to `history_length` timesteps.

        Args:
            idx (int): The current index for which to retrieve the history.

        Returns:
            torch.Tensor: Concatenated history of states.
        """
        history = torch.empty((0,))
        if self.history_length > 0:
            history = self.state_history[self.history_indices[idx]].reshape(-1)
        return history

    def __getitem__(self, idx):
        """
        Get the data for a single timestep.

        Args:
            idx (int): Index for the timestep.

        Returns:
            dict: Dictionary containing concatenated tensors of data.
        """
        # Retrieve the history of states
        state_history = self._get_history(idx)

        # Inputs
        if self.contact_conditioned:
            inputs = torch.cat((
                self.state[idx],
                state_history,
                self.target_contact_b[idx],
                self.feet_contact[idx],
                self.time_to_contact[idx],
            ))
        else:  # Velocity conditioned
            inputs = torch.cat((
                self.state[idx],
                state_history,
                self.v_des[idx],
                self.w_des[idx],
                self.gait_phase[idx],
            ))

        # Add noise to the inputs
        if self.sigma_noise > 0.:
            noise = torch.randn_like(inputs) * self.sigma_noise
            inputs += noise

        # Targets
        if self.use_set_points:
            targets = self.pd_set_points[idx]
        else:
            targets = self.tau[idx]

        if self.CVAE:
            item = {
                "input" : targets,
                "condition" : inputs,
            }
        else:
            item = {
                "input": inputs,
                "target": targets,
            }
            
        return item
    
    def __len__(self):
        return self.N
    
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