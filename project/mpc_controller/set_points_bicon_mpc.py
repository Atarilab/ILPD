import time
import numpy as np
import pinocchio as pin
from scipy.linalg import expm
from typing import Any

from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.torque_set_points import TorqueSetPointsBiconMPC

class PDTorquesBiconMPC(BiConMPC):
    def __init__(self,
                 robot: PinQuadRobotWrapper,
                 Kp:float = 3.,
                 Kd:float = 0.3,
                 use_mpc_torques = True,
                 **kwargs) -> None:
        super().__init__(robot, **kwargs)
        
        self.torque_set_points = TorqueSetPointsBiconMPC(self, Kp, Kd)
        self.use_mpc_torques = use_mpc_torques
        
    def get_torques(self, q: np.ndarray, v: np.ndarray, robot_data: Any) -> dict[float, Any]:
        """
        Returns torques from simulation data.

        Args:
            q (np.array): position state (nq)
            v (np.array): velocity state (nv)
            robot_data (MjData): MuJoco simulation robot data

        Returns:
            dict[float]: torque command {joint_name : torque value}
        """
        sim_t = round(robot_data.time, 3)

        # Replanning
        if self.pln_ctr == 0:
            pr_st = time.time()

            self.robot.update(q, v)

            # Contact plan in world frame
            self.mpc_cnt_plan_w = self.get_desired_contacts(q, v)
            self.xs_plan, self.us_plan, self.f_plan = self.gait_gen.optimize(
                q,
                v,
                sim_t,
                self.v_des,
                self.w_des,
                cnt_plan_des=self.mpc_cnt_plan_w)
            
            if not self.use_mpc_torques:
                # Compute torque set points for the given plan
                self.tau_set_points = self.torque_set_points.get(q, v)
            
            pr_et = time.time() - pr_st
            self.index = 0
            
        # Second loop onwards lag is taken into account
        if (self.step > 0 and
            self.sim_opt_lag and
            self.step >= int(self.replanning_time/self.sim_dt)
            ):
            lag = int((1/self.sim_dt)*(pr_et - pr_st))
            self.index = lag
            
        # If no lag (self.lag < 0)
        elif (not self.sim_opt_lag and
              self.pln_ctr == 0 and
              self.step >= int(self.replanning_time/self.sim_dt)
              ):
            self.index = 0
            
        if self.use_mpc_torques:
            # Compute MPC torques
            tau = self.robot_id_ctrl.id_joint_torques(
                q,
                v,
                self.xs_plan[self.index][:self.robot.nq],
                self.xs_plan[self.index][self.robot.nq:],
                self.us_plan[self.index],
                self.f_plan[self.index])
        else:
            tau = self.torque_set_points.torques(self.tau_set_points, q, v)
            
        # Create command {joint_name : torque value}
        torque_command = {
            joint_name: tau[id]
            for joint_name, id
            in self.robot.joint_name2act_id.items()
        }

        # Increment timing variables
        self._step()
        
        return torque_command