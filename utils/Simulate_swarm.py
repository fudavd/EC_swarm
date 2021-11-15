"""
Loading and testing
"""
from isaacgym import gymapi
from isaacgym import gymutil
from scipy.spatial.transform import Rotation as R

import numpy as np

from calculate_fitness import FitnessCalculator  # Fitness calculator class, all functions are implemented as different
# methods of this class
from sensors import Sensors  # Sensor class, all types of sensors are are implemented as different
# methods of this class

from . import Controllers

def simulate_swarm(life_timeout: float, individual: Controllers.Controller, headless: bool) -> np.array:
    """
    Simulate the robot in isaac gym
    :param individual: robot controller for every member of the swarm
    :param life_timeout: how long should the robot live
    :param headless: Start UI for debugging
    :return: fitness of the individual
    """
    # %% Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="Loading and testing")

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 2

    # defining axis of rotation!
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = False

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # Create viewer
    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")

    # %% Initialize environment
    print("Initialize environment")
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 0
    plane_params.dynamic_friction = 0
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.0001

    # Set up the env grid
    num_envs = 1
    spacing = 25.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Some common handles for later use
    print("Creating %d environments" % num_envs)

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_envs)
    # %% Initialize robot: Robobo
    print("Initialize Robot")
    # Load robot asset
    asset_root = "./"
    robot_asset_file = "/models/thymio/model.urdf"

    num_robots = 14
    distance = 0.5
    robot_handles = []
    initial_positions = np.zeros((2, num_robots))  # Allocation to save initial positions of robots

    controller_update_time = 0.2
    rows = 4
    # place robots
    # assert (pop_size%num_robots==0)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.032)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)
    for i in range(num_robots):
        pose.p = gymapi.Vec3(
            (((i + 9) // 12 + i) % rows) * distance - (rows - 1) / 2 * distance,
            (((i + 9) // 12 + i) // rows) * distance - (rows - 1) / 2 * distance,
            0.033)
        initial_positions[0][i] = pose.p.x  # Save initial position x of i'th robot
        initial_positions[1][i] = pose.p.y  # Save initial position y of i'th robot

        print("Loading asset '%s' from '%s', #%i" % (robot_asset_file, asset_root, i))
        robot_asset = gym.load_asset(
            sim, asset_root, robot_asset_file, asset_options)

        # add robot
        robot_handle = gym.create_actor(env, robot_asset, pose, f"robot_{i}", 0, 0)
        robot_handles.append(robot_handle)

    # get joint limits and ranges for robot
    props = gym.get_actor_dof_properties(env, robot_handle)

    # Give a desired velocity to drive
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    props["stiffness"].fill(0)
    props["damping"].fill(10)
    velocity_limits = props["velocity"]
    for i in range(num_robots):
        robot_handle = robot_handles[i]
        shape_props = gym.get_actor_rigid_shape_properties(env, robot_handle)
        shape_props[3].friction = 0
        shape_props[3].restitution = 1
        gym.set_actor_dof_properties(env, robot_handle, props)
        gym.set_actor_rigid_shape_properties(env, robot_handle, shape_props)

    # Point camera at environments
    cam_pos = gymapi.Vec3(-2, 0, 2)
    cam_target = gymapi.Vec3(0, 0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # # subscribe to spacebar event for reset
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

    def update_robot(sensor_input_distance, sensor_input_heading):
        for i in range(num_robots):
            # state : np.array() --> 5 by 1. [0:3] --> distance sensor output, [4] --> heading sensor output
            state = np.array([sensor_input_distance[i, :], sensor_input_heading[i]])
            velocity_target = individual.velocity_commands(state)  # assumed to be in format of [u,w]
            n_l = (velocity_target[0] - (velocity_target[1] / 2) * 0.085) / 0.021
            n_r = (velocity_target[0] + (velocity_target[1] / 2) * 0.085) / 0.021
            gym.set_actor_dof_velocity_targets(env, robot_handles[i], [n_l, n_r])

    def get_pos_and_headings():
        headings = []
        positions_x = []
        positions_y = []

        for i in range(num_robots):
            body_angle = gym.get_actor_rigid_body_states(env, robot_handles[i], gymapi.STATE_POS)["pose"]["r"][0]
            body_angle_mat = np.array([body_angle[0], body_angle[1], body_angle[2], body_angle[3]])
            r = R.from_quat(body_angle_mat)
            r = r.as_euler('zyx')
            headings.append(r[0])

            position = gym.get_actor_rigid_body_states(env, robot_handles[i], gymapi.STATE_POS)["pose"]["p"][0:1]
            positions_x.append(position[0][0])
            positions_y.append(position[0][1])

            return (np.array(headings), np.array(positions_x), np.array(positions_y))

    t = 0

    desired_movement = 10  # This values is required for the "movement" metric
    timestep = 0  # Counter to save total time steps, required for final step of fitness value calculation

    positions = np.full([3, num_robots], 0.01)  # Allocation to save positions of all robots in a single matrix
    fitness_calculator = FitnessCalculator(num_robots, initial_positions, desired_movement)  #  Fitness calculator init
    sensor = Sensors()  # Sensors init

    # %% Simulate
    while t <= life_timeout:
        # Every 0.01 seconds the velocity of the joints is set
        t = gym.get_sim_time(sim)

        if round(t, 3) % controller_update_time == 0.0:
            headings, positions[0], positions[1] = get_pos_and_headings()  # Update positions and headings of all robots

            distance_sensor_outputs = sensor.four_dir_sensor(positions, headings)  # The values recorded by on-board
            # sensors of robots
            heading_sensor_outputs = sensor.heading_sensor(positions, headings)  # The values recorded by on-board
            # sensors of robots
            update_robot(distance_sensor_outputs, heading_sensor_outputs)  # The connection to robot controller.
            # Ideally, this function accepts sensor readings, evaluates them on a controller (NN) and applies actuation
            # commands for each robot.

            timestep = timestep + 1  # Time step counter
            fitness_coh_and_sep = fitness_calculator.calculate_cohesion_and_separation(positions)  # Update fitness val
            fitness_alignment = fitness_calculator.calculate_alignment(headings)  # Update fitness val
            fitness_movement = fitness_calculator.calculate_movement(positions)  # Update fitness val

            # nogs = fitness_calculator.calculate_number_of_groups(positions)  # Update fitness val

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    experiment_fitness = fitness_coh_and_sep[0] / timestep - fitness_coh_and_sep[
        1] / timestep + fitness_alignment / timestep + fitness_movement  # For the time average, divide by time step
    return experiment_fitness
