"""
Loading and testing
"""
from typing import AnyStr

from isaacgym import gymapi
from isaacgym import gymutil
from uuid import uuid4
import math
import numpy as np
from . import Controllers
import os


def simulate_swarm(life_timeout: float, individual, headless: bool):
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
    sim_params.dt = 1.0 / 100
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
        sim_params.physx.use_gpu = True

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
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    # Set up the env grid
    num_envs = 1
    spacing = 25.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Some common handles for later use
    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.032)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

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

    controller_update_time = 0.25
    rows = 4
    # place robots
    # assert (pop_size%num_robots==0)

    for i in range(num_robots):
        pose.p = gymapi.Vec3(
            (((i + 9) // 12 + i) % rows) * distance - (rows-1) / 2 * distance,
            (((i + 9) // 12 + i) // rows) * distance - (rows-1) / 2 * distance,
            0.032)

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
    props["stiffness"].fill(1000.0)
    props["damping"].fill(600.0)
    robot_num_dofs = len(props)

    # Point camera at environments
    cam_pos = gymapi.Vec3(-4.0, -0.0, 4.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # # subscribe to spacebar event for reset
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    # # create a local copy of initial state, which we can send back for reset
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

    # def get_controller_input(robot_handle):
    #     current_sim_state = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)
    #     return controller_input

    def update_robot():
        # gym.clear_lines(viewer)
        for i in range(num_robots):
            robot_handle = robot_handles[i]

            position_target = individual.controller()
            gym.set_actor_dof_position_targets(env, robot_handle, position_target)

    def obtain_fitness(env, body):
        body_states = gym.get_actor_rigid_body_states(env, body, gymapi.STATE_POS)["pose"]["p"][0]
        current_pos = np.array((body_states[0], body_states[1], body_states[2]))
        # pose0 = initial_state["pose"]["p"][0]
        pose0 = gymapi.Vec3((body % rows) * distance, (body // rows) * distance, 0.032)
        absolute_distance = current_pos[1] - pose0.y
        return absolute_distance

    t = 0
    # %% Simulate
    while t <= life_timeout:
        # Every 0.01 seconds the velocity of the joints is set
        t = gym.get_sim_time(sim)

        if t % controller_update_time == 0.0:
            update_robot()

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    return
