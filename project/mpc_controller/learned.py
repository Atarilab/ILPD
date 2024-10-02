import copy
from typing import Any, Dict, Tuple
import numpy as np
import pinocchio as  pin 
import torch
import os
import pickle

from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.mpc_data_recorder import MPCDataRecorder
from mpc_controller.torque_set_points import TorqueSetPointsBiconMPC
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from learning.utils.model_utils import get_model_and_config, get_normalization_stats
from learning.data.MPCTrajectoryDataset import MPCTrajectoryDataset
from utils.visuals import express_contact_plan_in_consistant_frame


class LearnedBiconMPC(BiConMPC):

    NORMALIZATION_FILE = "normalization_stats.pkl"

    def __init__(self,
                 robot_wrapper: MJPinQuadRobotWrapper,
                 model_path: str = "",
                 policy_freq: int = 1000,
                 Kp: float = 3.,
                 Kd: float = 0.3,
                 **kwargs) -> None:
        super().__init__(robot_wrapper.pin, **kwargs)
        
        # Mujoco robot for contacts
        self.mj_robot = robot_wrapper.mj
        self.policy_freq = min(policy_freq, int(1 / self.sim_dt))
        self.policy_step_period = 1. / (self.policy_freq * self.sim_dt)
        self.Kp = Kp
        self.Kd = Kd
        
        # Initialize variables
        self.next_cnt_pos_w = np.empty((4, 3))
        self.next_cnt_abs_time = np.empty((4))
        self.nu = self.robot.nu
        self.model_output = np.zeros(self.nu)
        self.torque_set_points = TorqueSetPointsBiconMPC(self, Kp, Kd)
        self.feet_name = ["FL", "FR", "RL", "RR"]
        
        # State history buffer
        self.state_history = []  # Buffer to hold the recent state history

        # Load model and config in run dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model, cfg = get_model_and_config(model_path)
        self.model = self.model.to(self.device).eval()
        
        # Get config parameters
        cfg_dict = cfg.get_cfg_as_dict()
        self.use_set_points = cfg_dict.get("use_set_points", False)
        self.contact_conditioned = cfg_dict.get("contact_conditioned", False)
        self.history_length = cfg_dict.get("history_length", 0)
        self.state_variables = cfg_dict.get("state_variables", [])
        self.history_variables = cfg_dict.get("history_variables", [])
        self.var_to_normalize = MPCTrajectoryDataset.VARIABLE_TO_NORMALIZE
        
        if not self.state_variables:
            self.state_variables = MPCTrajectoryDataset.DEFAULT_STATE_VARIABLES
        if not self.history_variables:
            self.history_variables = MPCTrajectoryDataset.DEFAULT_STATE_VARIABLES
            
        if self.use_set_points:
            self.joint_name2act_id = self.robot.joint_name2act_id
        else:
            self.joint_name2act_id = self.mj_robot.joint_name2act_id

        # Load or initialize normalization statistics
        self.normalize = cfg_dict.get("normalize", False)
        if self.normalize:
            self.norm_stats = get_normalization_stats(model_path)
            self.mean = self.norm_stats["mean"]
            self.std = self.norm_stats["std"]
            # Convert to numpy array
            for d in [self.mean, self.std]:
                for k, v in d.items():
                    d[k] = v.numpy()
            
            if self.use_set_points:
                self.mean_target = self.mean["pd_set_points"]
                self.std_target = self.std["pd_set_points"]
            else:
                self.mean_target = self.mean["tau"]
                self.std_target = self.std["tau"]
        else:
            print("No statistics file provided, skipping normalization.")

            
    def load_data_stats(self):
        """
        Load normalization statistics (mean and std) from a file.
        """
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'rb') as f:
                self.data_stats = pickle.load(f)
                if self.contact_conditioned:
                    mean_input = self.data_stats['mean']
                    std_input = self.data_stats['std']
                    mean_input[-5:] = 0. 
                    std_input[-5:] = 1.
                else:    
                    mean_input = self.data_stats['mean'][:15+18+12+5]
                    std_input = self.data_stats['std'][:15+18+12+5]
                    # v_des, w_des and gait phase are not normalized
                    mean_input[-5:] = 0. 
                    std_input[-5:] = 1.
                    
                self.mean_input = mean_input.numpy()
                self.std_input = std_input.numpy()
                
                # Targets
                if not self.use_set_points:
                    self.mean_target = self.data_stats['mean_tau'].numpy()
                    self.std_target = self.data_stats['std_tau'].numpy()
                else:
                    self.mean_target = self.data_stats['mean_setpoints'].numpy()
                    self.std_target = self.data_stats['std_setpoints'].numpy()

                print(f"Loaded normalization statistics from {self.stats_file}")
        else:
            print(f"Statistics file {self.stats_file} not found, skipping normalization.")

    def normalize_inputs(self, variables : dict) -> dict:
        """
        Normalize the input using the precomputed mean and standard deviation. 
        This handles inputs with concatenated state history.
        """
        normalized_variable = {}
        for var_name, value in variables.items():
            if var_name in self.var_to_normalize:
                command = f"normalized_variable['{var_name}'] = (value - self.mean['{var_name}']) / (self.std['{var_name}'] + 1.0e-8)"
                exec(command)
        
        return normalized_variable
    
    def denormalize_targets(self, targets: np.ndarray) -> np.ndarray:
        targets = targets * self.std_target + self.mean_target
        return targets

    @staticmethod
    def transform_points(b_T_W, points_w) -> np.ndarray:
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_w.shape[0], 1))
        points_w_homogeneous = np.hstack((points_w, ones))
        points_b_homogeneous = b_T_W @ points_w_homogeneous.T
        points_b = points_b_homogeneous[:3, :].T
        return points_b

    def _store_state_history(self, current_state : np.ndarray):
        """
        Store the current state in the history buffer.
        The history include the current state.
        Args:
            current_state (np.ndarray): current state (normalized already)
        """
        self.state_history.insert(0, current_state)

        # If the buffer exceeds the history length, remove the oldest state
        if len(self.state_history) > self.history_length + 1:
            self.state_history.pop()

    def _get_state_history(self) -> np.ndarray:
        """
        Retrieve the concatenated history of the previous `history_length` states.
        If the history is shorter than the requested length (e.g., at the start of a run), 
        the current state is duplicated to fill the gap.
        """
        if len(self.state_history) < self.history_length + 1:
            # If the history is shorter, pad by repeating the last available state
            padding = [self.state_history[-1]] * (self.history_length + 1 - len(self.state_history))
            # Exclude the first state (current state)
            return np.concatenate(self.state_history[1:] + padding)
        else:
            # Exclude the first state (current state)
            return np.concatenate(self.state_history[1:])
        
    def get_inputs(self,
                   q,
                   v,
                   mj_data) -> np.ndarray:
        """
        Return model input data as np.array, normalized if statistics are available.
        Includes state history as input.
        """
        t = mj_data.time
        
        #
        pose, qj = np.split(q, [7], axis=-1)
        
        # Feet position in base frame
        feet_pos_b = self.mj_robot.get_foot_pos_base().reshape(-1)

        # Foot in contact
        contacts = self.mj_robot.foot_contacts()
        feet_contact = np.array([contacts[foot_name] for foot_name in self.feet_name])

        # Gravity in base frame
        B_T_W = pin.XYZQUATToSE3(pose).inverse()
        B_R_W = B_T_W.rotation
        gravity_b = -B_R_W[:, -1]
        
        # Contacts position and timings
        time_to_cnt = np.round(np.clip(self.next_cnt_abs_time - t, 0., np.inf), 3)
        target_contact_b = self.transform_points(B_T_W, self.next_cnt_pos_w).reshape(-1)

        # Gait phase
        phase = self.gait_gen.gait_planner.get_phi(t, 0) / self.gait_period
        
        # Normalize variables
        del contacts, B_T_W, B_R_W, t, pose
        if self.normalize:
            normalized_variables = self.normalize_inputs(locals())
            for var_name in normalized_variables.keys():
                command = f"{var_name} = normalized_variables['{var_name}']"
                exec(command)
            
        # Concatenate inputs array
        inputs = []
        # state
        for var_name in self.state_variables:
            exec(f"inputs.append({var_name})")
        state_t = np.concatenate(inputs)
        
        # Add state to history
        self._store_state_history(state_t)

        # history
        state_history = self._get_state_history()
        inputs.append(state_history)
        
        # goal conditioning
        if self.contact_conditioned:
            inputs.append(target_contact_b)
            inputs.append(time_to_cnt)
            
        else:
            inputs.append(self.v_des)
            inputs.append([self.w_des])
            inputs.append([phase])

        inputs_array = np.concatenate(inputs)
        return inputs_array
    
    def _call_policy(self) -> bool:
        return self.step % self.policy_step_period == 0
    
    def get_torques(self,
                    q: np.ndarray,
                    v: np.ndarray,
                    robot_data: Any,
                    ) -> dict[float]:
        """
        Returns torques from simulation data.

        Args:
            q (np.array): position state (nq)
            v (np.array): velocity state (nv)
            robot_data (MjData): MuJoco simulation robot data

        Returns:
            dict[float]: torque command {joint_name : torque value}
        """
        q_copy = q.copy()
        v_copy = v.copy()
        self.robot.update(q_copy, v_copy)

        t = robot_data.time
        
        # Contacts (updated at planner rate)
        if self.contact_conditioned and self.pln_ctr == 0:
            cnt_plan = self.gait_gen.compute_raibert_contact_plan(q_copy, v_copy, t, self.v_des, self.w_des)
            self.gait_gen.cnt_plan = cnt_plan
            is_cnt_plan, cnt_plan_pos = np.split(cnt_plan, [1], axis=-1)
            cnt_plan_pos_w = express_contact_plan_in_consistant_frame(q_copy, cnt_plan_pos, base_frame=False)
            next_cnt_t_index = np.argmax(is_cnt_plan > 0, axis=0).reshape(-1)
            self.next_cnt_pos_w = cnt_plan_pos_w[next_cnt_t_index, np.arange(len(next_cnt_t_index)), :]
            gait_dt = self.gait_gen.params.gait_dt
            self.next_cnt_abs_time = next_cnt_t_index * gait_dt + t
            self.index = 0
                
        if self._call_policy():
            inputs_array = self.get_inputs(q_copy, v_copy, robot_data)
            inputs_tensor = torch.from_numpy(inputs_array).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                self.model_output = self.model(inputs_tensor).squeeze().cpu().numpy()
            
            if self.normalize:
                self.model_output = self.denormalize_targets(self.model_output)

        if self.use_set_points:
            tau_output = self.torque_set_points.torques(self.model_output, q_copy, v_copy)
        else:
            tau_output = self.model_output

        torque_command = {
            joint_name: tau_output[id]
            for joint_name, id in self.joint_name2act_id.items()
        }

        self._step()
        return torque_command
