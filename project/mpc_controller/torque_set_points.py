import numpy as np
import pinocchio as pin

from mpc_controller.bicon_mpc import BiConMPC
from scipy.linalg import expm


class TorqueSetPointsBiconMPC():
    """
    Computes torques set points of a given BiConMPC controller.
    """
    def __init__(self,
                 controller : BiConMPC,
                 Kp : float = 3.,
                 Kd : float = 0.5) -> None:
        self.controller = controller
        self.pin_robot = controller.robot
        self.eff_arr = [self.pin_robot.frame_name2id[name] for name in self.pin_robot.foot_names]

        # Gains of the PD torque controller
        self.n = self.pin_robot.nv - 6 # Only joints
        self.T = self.controller.horizon * self.controller.sim_dt
        self.offset_end_plan = 1

        self.Kp = np.ones(self.n) * Kp
        self.Kd = np.ones(self.n) * Kd
        
        # Scale gains for elbow and shoulder joints
        self.Kd[::3] *= 8.
        self.Kp[2::3] *= 4.
        
    @staticmethod
    def find_C(A, B, q0, dq0, qT, dqT, T):
        """
        Finds the optimal C so that the error at time T is minimized.

        Args:
        - A: Matrix A of the system
        - B: Matrix B of the system
        - q0: Initial position
        - dq0: Initial velocity
        - qT: Target position at final time T
        - dqT: Target velocity at final time T
        - T: Final time
        
        Returns:
        - C_optimal: Optimal constant vector C
        """
        n = A.shape[0]
        A_inv = np.linalg.inv(A)
        
        # Precompute the system matrix M
        M = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-A_inv, -A_inv @ B]
        ])
        
        yT = np.concatenate([qT, dqT])  # Initial condition as [q0, dq0]
        y0 = np.concatenate([q0, dq0])  # Initial condition as [q0, dq0]

        eM_T = expm(M * T)
        M_2 = yT - eM_T @ y0
        M_1 = np.eye(2*n) - eM_T
        C_opt = (np.linalg.inv(M_1) @ M_2)[:n]
        
        return C_opt
        
    def compute_dyn_drift_term(self,
                               q : np.array,
                               dq : np.array,
                               f_ext : np.array) -> None:
        """
        Compute drift term b in dynamic equation.
        tau = M * ddq + b
        b = C(q, dq) - tau_ext(q, f_ext)
        """
        a0 = np.zeros_like(dq)
        v0 = np.zeros(3)
        C_dyn = pin.rnea(self.pin_robot.model, self.pin_robot.data, q, dq, a0)

        # Initialize effective torque array
        tau_ext = np.zeros(self.pin_robot.nv)
        for eff_id, f in zip(self.eff_arr, f_ext.reshape(-1, 3)):
            # Compute Jacobian transpose for the current end-effector
            J = pin.computeFrameJacobian(self.pin_robot.model, self.pin_robot.data, q, eff_id, pin.LOCAL_WORLD_ALIGNED).T
            # Compute and accumulate the effective torques
            tau_ext += np.matmul(J, np.hstack((f, v0)))

        b = C_dyn - tau_ext
        return b
    
    def compute_inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inertia matrix in the given configuration

        Args:
            q (np.ndarray): _description_
        """
        M = pin.crba(self.pin_robot.model, self.pin_robot.data, np.array(q))
        return M
    
    def get(self,
            q: np.ndarray,
            dq: np.ndarray) -> np.ndarray:
        """
        Compute pd torque set points for current MPC plan. 

        Args:
            q (np.ndarray) : position state
            dq (np.ndarray) : velocity state
            
        Returns:
            np.ndarray: pd control points for the torques
        """
        # Compute dynamic terms in the current configuration
        M = self.compute_inertia_matrix(q)[-self.n:, -self.n:]
        f_des = np.median(self.controller.f_plan, axis=0)
        C_dyn = self.compute_dyn_drift_term(q, dq, f_des)[-self.n:]

        # ODE: M.ddq/ddt + Kd.dq/dt + Kp.q(t) + C = 0
        # with C = - C_dyn + Kp.tau_setpoint
        # -> A.ddq/ddt + B.dq/dt + q(t) + C_ = 0
        A = M / self.Kp
        B = np.eye(self.n) * self.Kd / self.Kp

        # Find C such that q(t=T) ~ q_des and dq(t=T) ~ dq_des
        q0 = q[-self.n:]
        dq0 = dq[-self.n:]
        qT = self.controller.xs_plan[-self.offset_end_plan, 7:self.n + 7]
        dqT = self.controller.xs_plan[-self.offset_end_plan, -self.n:]
        C_optimal = self.find_C(A, B, q0, dq0, qT, dqT, self.T)
        
        # Compute set point according to optimal C value
        tau_set_points = C_dyn / self.Kp + C_optimal
        
        return tau_set_points
    
    def torques(self,
                tau_set_points: np.ndarray,
                q: np.ndarray,
                v: np.ndarray) -> np.ndarray:
        """
        Compute torque values for the given set points.

        Args:
            torque_set_points (np.ndarray):
            q (np.ndarray):
            dq (np.ndarray):

        Returns:
            np.ndarray: torque values
        """
        tau = self.Kp * (tau_set_points - q[-self.n:]) - self.Kd * v[-self.n:]
        return tau