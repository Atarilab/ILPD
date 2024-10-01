import tyro
import numpy as np
import os

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.set_points_bicon_mpc import PDTorquesBiconMPC
from mpc_controller.learned import LearnedBiconMPC
from mpc_controller.mpc_data_recorder import MPCDataRecorder

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

def main(model_path : str = "",
         use_set_points : bool = False,
         gait : str = "trot",
         policy_freq : int = 50,
         Kp: float = 3.,
         Kd: float = 0.5,
         ):

    ###### Robot model
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )
    robot.pin.info()
    robot.mj.info()

    ###### Controller
    if not os.path.exists(model_path):
        controller = PDTorquesBiconMPC(
            robot.pin,
            replanning_time=0.05,
            sim_opt_lag=False,
            Kp=Kp,
            Kd=Kd,
            use_mpc_torques=not(use_set_points))
    else:
        controller = LearnedBiconMPC(
            robot,
            model_path,
            policy_freq,
            Kp=Kp,
            Kd=Kd,
        )
    # Set command
    v_des, w_des = np.array([0.3, 0., 0.]), 0
    controller.set_command(v_des, w_des)
    
    # Set gait
    # Choose between trot, jump and bound
    if gait == "trot":
        controller.set_gait_params(trot)  
    else:
        controller.set_gait_params(jump)

    ###### Simulator
    simulator = Simulator(robot.mj, controller)
    # Visualize contact locations
    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller))
    # Run simulation
    sim_time = 10 #s
    simulator.run(
        simulation_time=sim_time,
        use_viewer=True,
        real_time=False,
        visual_callback_fn=visual_callback,
        # force_intensity=20.,
        # force_duration=0.1,
        # force_period=0.75,
    )
        
if __name__ == "__main__":
    args = tyro.cli(main)