import os
import time
import tyro
import numpy as np
import multiprocessing as mp
from functools import partial

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.mpc_data_recorder import MPCDataRecorder
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.set_points_bicon_mpc import PDTorquesBiconMPC
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

EXT_FORCE_INTENSITY = 20
EXT_FORCE_DURATION = 0.1
EXT_FORCE_PERIOD = .75
SKIP_FIRST_SECONDS = 0.0
DEFAULT_DATA_DIR = "../data/"

def run_episode(episode_idx: int,
                record_dir: str,
                gait: str,
                n_points: int,
                record_step: int,
                Kp: float,
                Kd: float,
                random_velocity: bool,
                use_viewer: bool,
                use_perturbations: bool,
                use_set_points: bool):
    """
    Function to run a single episode in parallel across multiple cores.
    Saves the recorded data for the episode in a separate directory.
    """
    time.sleep(0.1 * episode_idx % 20)
    seed = int(episode_idx * time.time() * 77777 % 33333  + time.time() * 88888 % 33333)
    np.random.seed(seed)    

    ###### Robot model
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
    )

    ###### Controller
    controller = PDTorquesBiconMPC(
        robot.pin,
        replanning_time=0.05,
        sim_opt_lag=False,
        Kp=Kp,
        Kd=Kd,
        use_mpc_torques=not(use_set_points))
    
    # Set command
    if not random_velocity:
        v_des, w_des = np.array([0.3, 0., 0.]), 0
        controller.set_command(v_des, w_des)
    
    # Set gait
    if gait == "trot":
        controller.set_gait_params(trot)
    else:
        controller.set_gait_params(jump)
    
    ###### Data recorder
    gait_str = controller.gait_gen.params.motion_name
    episode_record_dir = os.path.join(record_dir, gait_str, str(episode_idx))
    os.makedirs(episode_record_dir, exist_ok=True)
    
    data_recorder = MPCDataRecorder(
        robot,
        controller,
        record_dir=episode_record_dir,
        record_step=record_step,
        Kp=Kp,
        Kd=Kd
    )

    ###### Simulator
    simulator = Simulator(robot.mj, controller, data_recorder)
    visual_callback = (lambda viewer, step, q, v, data:
                       desired_contact_locations_callback(viewer, step, q, v, data, controller))

    sim_time = n_points * simulator.sim_dt * record_step + SKIP_FIRST_SECONDS  # s
    use_pert = int(use_perturbations)

    robot.mj.reset_randomize()

    simulator.run(
        simulation_time=sim_time,
        use_viewer=use_viewer,
        real_time=False,
        stop_on_collision=True,
        visual_callback_fn=visual_callback,
        force_duration=EXT_FORCE_DURATION * use_pert,
        force_intensity=EXT_FORCE_INTENSITY * use_pert,
        force_period=EXT_FORCE_PERIOD * use_pert,
    )
    skip_last_s = 0.
    if robot.collided:
        skip_last_s = 1.
    
    # Save the data
    data_recorder.save(skip_first_s=SKIP_FIRST_SECONDS, skip_last_s=skip_last_s)
    
    return episode_record_dir

def aggregate_data(record_dirs, final_record_dir):
    """
    Aggregate the recorded data from each episode into a single directory.
    """
    os.makedirs(final_record_dir, exist_ok=True)

    # Assuming we are combining data files in `.npz` format
    combined_data = {}

    for record_dir in record_dirs:
        data_file = os.path.join(record_dir, "data.npz")
        if os.path.exists(data_file):
            data = np.load(data_file)
            for key in data.files:
                if key not in combined_data:
                    combined_data[key] = []
                combined_data[key].append(data[key])
        # Clean files
        # os.remove(data_file)
        # os.removedirs(record_dir)

    # Save the combined data
    final_data_file = os.path.join(final_record_dir, "data.npz")

    # Concatenate with existing data if exists
    if os.path.exists(final_data_file):
        existing_data = np.load(final_data_file)
        for key in existing_data.files:
            if key not in combined_data:
                combined_data[key] = []
            combined_data[key].append(data[key])
            
    # Concatenate the data for each key
    for key in combined_data:
        combined_data[key] = np.concatenate(combined_data[key], axis=0)

    np.savez(final_data_file, **combined_data)  
    print(f"Combined data saved to {final_data_file}")

def main(record_dir: str = DEFAULT_DATA_DIR,
         gait: str = "trot",
         n_points: int = 5000,
         record_step: int = 2,
         episodes: int = 10,
         Kp: float = 3.,
         Kd: float = 0.5,
         random_velocity: bool = False,
         use_viewer: bool = False,
         use_perturbations: bool = True,
         use_set_points: bool = False,
         num_workers: int = 1):
    
    ###### Run episodes in parallel using multiprocessing
    with mp.Pool(processes=num_workers) as pool:
        episode_func = partial(
            run_episode,
            record_dir=record_dir,
            gait=gait,
            n_points=n_points,
            record_step=record_step,
            Kp=Kp,
            Kd=Kd,
            random_velocity=random_velocity,
            use_viewer=use_viewer,
            use_perturbations=use_perturbations,
            use_set_points=use_set_points
        )
        # Parallel execution of episodes
        record_dirs = pool.map(episode_func, range(episodes))

    ###### Aggregate results from all episodes
    final_record_dir = os.path.join(record_dir, gait)
    aggregate_data(record_dirs, final_record_dir)

if __name__ == "__main__":
    args = tyro.cli(main)
