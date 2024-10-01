import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mujoco._structs import MjData

UPDATE_VISUALS_STEPS = 50 # Update position every <UPDATE_VISUALS_STEPS> sim steps
FEET_COLORS = [
    [1., 0., 0., 1.], # FR
    [0., 1., 0., 1.], # FL
    [0., 0., 1., 1.], # RR
    [1., 1., 1., 1.], # RL
]
N_NEXT_CONTACTS = 12
SPHERE_RADIUS = 0.012

def express_contact_plan_in_consistant_frame(q : np.ndarray,
                                            cnt_plan_pos : np.ndarray,
                                            base_frame : bool = False) -> np.ndarray:
    """
    Express contact plan positions in the same frame.
    Gait generator gives contact plan with x, y in base frame and z in world frame.

    Args:
        - q (np.ndarray): state of the robot
        - cnt_plan_pos (np.ndarray): 3D positions of the contact plan.
        x, y are in base frame while z is in world frame.
        shape [H, 4 (feet), 3]
        - base_frame (bool): express contact plan in base frame. Otherwise world frame.

    Returns:
        np.ndarray: cnt_plan_pos in base frame
    """
    # For all points of the cnt plan
    # [x_B, y_B, z_B, 1].T = B_T_W . [x_W, y_W, z_W, 1].T
    # z_B, x_W and y_W are unknown
    # One can express the equality as A.X = B with:
    # X = [x_W, y_W, z_B]
    # A = [[1,0,0], [0,1,0], [0,0,0]] - B_R_W[:, -1]
    # B = W_R_B . ([x_B, y_B, 0].T - W_p_B) - [0, 0, z_W].T
    # (W_p_B = [0., 0., 0.].T as the contact plan is computed with the base at the origin)
    # Then X = A^{-1}.B
 
    # Reshape to process all positions at once
    cnt_plan_p = cnt_plan_pos.reshape(-1, 3).copy()

    # Rotation matrix W_R_B from world to base
    W_R_B = pin.Quaternion(q[3:7]).matrix()  # Rotation matrix from base to world
    W_p_B = q[:3]  # Translation vector from base to world
    
    # Analytical form of the inverse of A
    A_inv = np.diag([1., 1., 1. / W_R_B[-1, -1]])
    A_inv[-1, :2] = - W_R_B[:2, -1] / W_R_B[-1, -1]

    # Prepare the contact positions for vectorized operation
    p_B = cnt_plan_p.copy()
    p_B[:, -1] = 0.  # Set z_B to 0 for all points

    # Compute B for all contact points in one operation
    B = W_R_B @ p_B.T  # Apply rotation to the base frame points
    B = B.T  # Transpose to get shape [N, 3]
    B[:, -1] -= cnt_plan_p[:, -1]  # Subtract z_W from the last coordinate of B

    # Compute X for all positions at once
    X = (A_inv @ B.T).T

    # Apply the final transformations based on the base_frame flag
    if base_frame:
        cnt_plan_p[:, -1] = X[:, -1]  # Update z_B in base frame
    else:
        cnt_plan_p[:, :-1] = X[:, :-1] + W_p_B[:-1]  # Update x_W and y_W in world frame

    # Reshape back to the original shape [H, 4, 3]
    cnt_plan_p = cnt_plan_p.reshape(-1, 4, 3)

    return cnt_plan_p
    
def desired_contact_locations_callback(viewer,
                                       sim_step: int,
                                       q: np.ndarray,
                                       v: np.ndarray,
                                       robot_data: MjData,
                                       controller: ControllerAbstract) -> None:
    """
    Visualize the desired contact plan locations in the MuJoCo viewer.
    """
    
    if UPDATE_VISUALS_STEPS % UPDATE_VISUALS_STEPS == 0: 
        
        # Next contacts in base frame (except height in world frame)
        horizon_step = controller.gait_gen.horizon
        contact_step = max(horizon_step // N_NEXT_CONTACTS, 1)
        next_contacts_B = controller.gait_gen.cnt_plan[::contact_step, :, 1:]
        
        contact_plan_W = express_contact_plan_in_consistant_frame(q, next_contacts_B, base_frame=False).reshape(-1, 3)
        
        viewer.user_scn.ngeom = 0
        
        for i, contact_W in enumerate(contact_plan_W):
            color = FEET_COLORS[i % len(FEET_COLORS)]
            color[-1] = 0.4 if i > 4 else 1.
            size = SPHERE_RADIUS if i < 4 else SPHERE_RADIUS / 2.
            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=contact_W,
                mat=np.eye(3).flatten(),
                rgba=color,
            )
        
        viewer.user_scn.ngeom = i + 1