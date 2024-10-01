import os
import time
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.torque_set_points import TorqueSetPointsBiconMPC
from utils.visuals import express_contact_plan_in_consistant_frame

### Data recorder
class MPCDataRecorder(DataRecorderAbstract):
    FILE_NAME = "data.npz"
    STATE_Q = "q"
    STATE_V = "v"
    FEET_POS = "feet_pos_w"
    TARGET_CONTACT = "target_cnt_w"
    TIME_TO_CONTACT = "time_to_cnt"
    FEET_CONTACT = "foot_cnt"
    EXTERNAL_FORCES = "f_ext"
    TAU = "tau"
    PD_SET_POINTS = "pd_set_points"
    V_DES = "v_des"
    W_DES = "w_des"
    GAIT_PHASE = "gait_phase"
    TIME = "time"
    
    def __init__(self,
                 robot: MJPinQuadRobotWrapper,
                 controller: BiConMPC,
                 record_dir: str = "",
                 record_step: int = 2,
                 Kp: float = 3.,
                 Kd: float = 0.3,
                 ) -> None:
        """
        MPCDataRecorder class.

        Args:
            robot (MJQuadRobotWrapper): Pin Robot Wrapper
            record_dir (str, optional): Directory to record data. Defaults to "".
            record_step (int, optional): Record data every <record_step>. Defaults to 2.
        """
        super().__init__(record_dir)
        
        self.mj_robot = robot.mj
        self.controller = controller
        self.pin_robot = controller.robot
        self.record_step = record_step
        self.torque_set_points = TorqueSetPointsBiconMPC(controller, Kp, Kd)
        self.tau_set_points = np.zeros(self.torque_set_points.n)
        self.gait_period = self.controller.gait_gen.params.gait_period
        self._update_record_dir(record_dir)
        
        self.keys = [
            ### Position state
            # [x, y, z, qx, qy, qz, qw, q0, ..., qJ] [19]
            MPCDataRecorder.STATE_Q,
            ### Velocity state
            # [v, w, dq0, ..., dqJ] [18]
            MPCDataRecorder.STATE_V,
            ### Foot position in world
            # [foot0_w, foot1_w, foot2_w, foot3_w] [4, 3]
            MPCDataRecorder.FEET_POS,
            ### Target contact locations in world
            # [target_cnt0_w, ..., target_cnt3_w] [4, 3]
            MPCDataRecorder.TARGET_CONTACT,
            ### Time to reach contact locations
            # [t0, t1, t2, t3] [4]
            MPCDataRecorder.TIME_TO_CONTACT,
            ### Is foot in contact (1: yes)
            # [is_cnt0, ... is_cnt3] [4]
            MPCDataRecorder.FEET_CONTACT,
            ### Feet external force
            # [f_0, ..., f_3] [4, 3]
            MPCDataRecorder.EXTERNAL_FORCES,
            ### Desired linear velocity
            # [vx, vy, vz] [3]
            MPCDataRecorder.V_DES,
            ### Desired angular velocity
            # [w_yaw] [1]
            MPCDataRecorder.W_DES,
            # Phase of the cyclic gait
            # [phase] [1]
            MPCDataRecorder.GAIT_PHASE,
            ### Torques
            # [tau_u0, ..., tau_uN] [12]
            MPCDataRecorder.TAU,
            ### PD control setpoints. tau = Kp(pd_q - q(t)) - Kd(dq(t))
            # [pd_q_u0, ..., pd_q_uN] [12]
            MPCDataRecorder.PD_SET_POINTS,
            ### Simulation time
            # [time] [1]
            MPCDataRecorder.TIME,
        ]

        self.reset()
        
    def _get_empty_data_dict(self) -> dict:
        d = {k : [] for k in self.keys}
        return d
        
    def _update_record_dir(self, record_dir:str) -> None:
        os.makedirs(record_dir, exist_ok=True)
        self.saving_file_path = os.path.join(record_dir, MPCDataRecorder.FILE_NAME)

    def reset(self) -> None:
        self.recorded_data = self._get_empty_data_dict()
        self.next_cnt_pos_w = np.zeros((4, 3))
        self.next_cnt_abs_time = np.zeros((4,))
        self.step = 0
            
    @staticmethod
    def transform_3d_points(A_T_B : np.ndarray, points_B : np.ndarray) -> np.ndarray:
        """
        Transforms an array of 3d points expressed in frame B
        to frame A according to the transform from B to A (A_T_B)

        Args:
            A_T_B (_type_): SE(3) transform from frame B to frame A
            p_B (_type_): 3D points expressed in frame B. Shape [N, 3]
        """
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_B.shape[0], 1))
        points_B_homogeneous = np.hstack((points_B, ones))
        # Apply the transformation matrix
        points_A_homogeneous = A_T_B @ points_B_homogeneous.T
        # Convert back to 3D coordinates
        points_A = points_A_homogeneous[:3, :].T
        return points_A
    
    def record(self, q: np.ndarray, v: np.ndarray, **kwargs) -> None:
        """ 
        Record data.
        Called by the simulator.
        """
        mj_data = kwargs.get("mj_data", None)
        t = mj_data.time
        q_copy = np.copy(q)
        v_copy = np.copy(v)
        self.pin_robot.update(q, v)

        # PD set points, update every mpc replanning
        # Target contacts, update contact plan positions when MPC is replanning
        if self.controller.pln_ctr - 1 == 0:
            self._update_controller_plan(q_copy, v_copy, t)
            
        if self.step % self.record_step == 0:
            # Make deep copies of q, v, and mj_data.ctrl to ensure values are not overwritten
            tau_copy = np.copy(mj_data.ctrl)
            
            # State
            self.recorded_data[MPCDataRecorder.STATE_Q].append(q_copy)
            self.recorded_data[MPCDataRecorder.STATE_V].append(v_copy)
            
            # Time
            self.recorded_data[MPCDataRecorder.TIME].append([t])
            
            # Torques
            self.recorded_data[MPCDataRecorder.TAU].append(tau_copy)


            self.recorded_data[MPCDataRecorder.PD_SET_POINTS].append(self.tau_set_points)
                
            # Desired velocity
            self.recorded_data[MPCDataRecorder.V_DES].append(self.controller.v_des)
            self.recorded_data[MPCDataRecorder.W_DES].append([self.controller.w_des])

            # Phase of the cyclic gate
            phase = self._get_gait_phase(time=t)
            self.recorded_data[MPCDataRecorder.GAIT_PHASE].append([phase])
            
            # Feet position world
            feet_pos_w = self.pin_robot.get_foot_pos_world()
            self.recorded_data[MPCDataRecorder.FEET_POS].append(feet_pos_w)
            
            # External force foot
            eeff_names = ["FL", "FR", "RL", "RR"]
            contact_forces = self.mj_robot.get_foot_contact_forces()
            contact_forces_array = np.array([contact_forces[eeff_name] for eeff_name in eeff_names])
            self.recorded_data[MPCDataRecorder.EXTERNAL_FORCES].append(contact_forces_array)

            # Foot in contact
            foot_contacts = self.mj_robot.foot_contacts()
            foot_contacts_array = np.array([foot_contacts[foot_name] for foot_name in eeff_names])
            self.recorded_data[MPCDataRecorder.FEET_CONTACT].append(foot_contacts_array)

            time_to_cnt = np.round(np.clip(self.next_cnt_abs_time - t, 0., np.inf), 3)
            self.recorded_data[MPCDataRecorder.TARGET_CONTACT].append(self.next_cnt_pos_w)
            self.recorded_data[MPCDataRecorder.TIME_TO_CONTACT].append(time_to_cnt)
            
        self.step += 1
        
    def _update_controller_plan(self,
                                q : np.ndarray,
                                v : np.ndarray,
                                time : float,
                                ) -> None:
        
        self.tau_set_points = self.torque_set_points.get(q, v)
    
        # shape [horizon, 4 (feet), 4 (cnt + pos)] 
        cnt_plan = self.controller.gait_gen.cnt_plan
        is_cnt_plan, cnt_plan_pos_b = np.split(cnt_plan, [1], axis=-1)
        cnt_plan_pos_w = express_contact_plan_in_consistant_frame(q, cnt_plan_pos_b, base_frame=False)

        # Next contact position
        next_cnt_t_index = np.argmax(is_cnt_plan>0, axis=0).reshape(-1)
        self.next_cnt_pos_w = cnt_plan_pos_w[next_cnt_t_index, np.arange(len(next_cnt_t_index)), :]
        
        # Absolute simulation time of the next contact
        gait_dt = self.controller.gait_gen.params.gait_dt
        self.next_cnt_abs_time = next_cnt_t_index * gait_dt + time
        
    def _get_gait_phase(self, time : float) -> float:
        """
        Return gait phase in [0, 1] 
        """
        phase = self.controller.gait_gen.gait_planner.get_phi(time, 0) / self.gait_period
        return phase

    def _append_and_save(self, skip_first, skip_last):
        """ 
        Append new data to existing file and save file.
        """
        N = len(self.recorded_data[MPCDataRecorder.STATE_Q])
        
        # If recording is long enough
        if N - skip_first - skip_last > 0:
            
            # Load data and append if exists
            if os.path.exists(self.saving_file_path):
                # Concatenate data
                data_file = np.load(self.saving_file_path)
                data = {k : data_file[k] for k in self.keys}
                if list(data.keys()) == list(self.recorded_data.keys()):
                    for k, v in self.recorded_data.items():
                        data[k] = np.concatenate(
                            (data[k], v[skip_first:N-skip_last]), axis=0
                        )
            else:
                data = self.recorded_data

            # Overrides the file with new data
            np.savez(self.saving_file_path, **data)

    def save(self,
             lock = None,
             skip_first_s : float = 0.,
             skip_last_s : float = 0.,
             ) -> None:
        # Avoid repetitve data and near failure states
        skip_first = int(skip_first_s * self.controller.sim_dt)
        skip_last = int(skip_last_s * self.controller.sim_dt)
        
        if lock:
            with lock:
                self._append_and_save(skip_first, skip_last)
        else:
            self._append_and_save(skip_first, skip_last)

        self.reset()