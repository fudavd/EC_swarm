import copy
import re
from multiprocessing import shared_memory, Process
from typing import AnyStr, TypedDict
from isaacgym import gymapi
from isaacgym import gymutil
from scipy.spatial.transform import Rotation as R

import numpy as np
from numpy.random import default_rng

from .Fitnesses import FitnessCalculator
from .Sensors import Sensors
from .plot_swarm import swarm_plotter


def calc_vel_targets(controller, states):
    velocity_target = controller.velocity_commands(np.array(states))
    n_l = ((velocity_target[0]+0.025) - (velocity_target[1] / 2) * 0.085) / 0.021
    n_r = ((velocity_target[0]+0.025) + (velocity_target[1] / 2) * 0.085) / 0.021
    return [n_l, n_r]


EnvSettings = {
    'arena_type': "circle_30x30",
    'spawn_radius': 1.0,
    'fitnesses': ['grad'],
    'objectives': ['alignment_sub', 'alignment', 'gradient_sub'],
    'record_video': False,
    'save_full_fitness': False
}


def simulate_swarm_population(life_timeout: float, individuals: list, swarm_size: int,
                              headless: bool,
                              arena: str = "circle_30x30",
                              env_params: TypedDict = EnvSettings) -> np.ndarray:
    """
    Parallelized simulation of all 'robot swarm' individuals in the population using Isaac gym
    
    :param life_timeout: how long should the robot live
    :param individuals: List of swarm instances that define every robot member phenotypes
    :param swarm_size: number of members inside the swarm
    :param headless: Start UI for debugging
    :param objectives: specify which components of the fitness function should be returned
    :param arena: Type of arena
    :param env_params: Dictionary of environment settings, if unspecified set to Default
    :return: fitness of the individual(s)
    """

    if_random_start = True  # omni, k_nearest, 4dir

    # %% Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="Loading and testing")

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.1
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

    # %% Initialize environment
    # print("Initialize environment")
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
    # asset_options.replace_cylinder_with_capsule = False
    asset_options.armature = 0.0001

    # Set up the env grid
    num_envs = len(individuals)
    spacing = int(re.findall('\d+', arena)[-1])
    env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Some common handles for later use
    desired_movement = 10  # This value is required for the "movement" metric
    env_list = []
    robot_handles_list = []
    subgroup_ratio_list = []
    fitness_list = []
    fitness_mat = []
    sensor_list = []
    num_robots = swarm_size
    controller_list = []
    controller_types_list = []
    print("Creating %d environments" % num_envs)
    for i_env in range(num_envs):
        individual = individuals[i_env]
        controller_list.append(np.array([member.controller for member in individual]))
        controller_types_list.append([member.controller.controller_type for member in individual])
        swarm_composition = [individual.count(i) for i in set(individual)]
        subgroup_ratio = swarm_composition[0]/len(individual)
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_envs)
        env_list.append(env)
        # %% Initialize robot: Robobo
        # print("Initialize Robot")
        # Load robot asset
        asset_root = "./"


        robot_handles = []
        initial_positions = np.zeros((2, num_robots))  # Allocation to save initial positions of robots

        if if_random_start:
            flag = 0
            init_area = 3.0 * np.sqrt(swarm_size/14)
            init_flag = 0
            init_failure_1 = 1
            rng = default_rng()

            radius_spawn = env_params['spawn_radius']
            # circle corner
            if arena.split('_')[:-1] == ['circle', 'corner']:
                 iangle = np.pi/2 * rng.random()
                 iy = ((15 + 12)*radius_spawn * (np.cos(iangle))) * spacing/30
                 ix = ((15 + 12)*radius_spawn * (np.sin(iangle))) * spacing/30
            # circle
            elif arena.split('_')[:-1] == ['circle']:
                print(f'loading with radius_spawn={radius_spawn}')
                iangle = np.pi*2 * rng.random()
                iy = (15 + 12*radius_spawn * (np.cos(iangle))) * spacing/30
                ix = (15 + 12*radius_spawn * (np.sin(iangle))) * spacing/30
            # linear
            elif arena.split('_')[:-1] == ['linear']:
                  iy = 27
                  ix = 15 + 2*radius_spawn*(rng.random()-0.5)

            a_x = ix + (init_area / 2)
            b_x = ix - (init_area / 2)
            a_y = iy + (init_area / 2)
            b_y = iy - (init_area / 2)

            while init_failure_1 == 1 and init_flag == 0:
                ixs = a_x + (b_x - a_x) * rng.random(num_robots)
                iys = a_y + (b_y - a_y) * rng.random(num_robots)
                flag = 0

                for i in range(num_robots):
                    for j in range(num_robots):
                        if i != j and np.sqrt(np.square(ixs[i] - ixs[j]) + np.square(iys[i] - iys[j])) < 0.4:
                            init_failure_1 = 1
                            flag = 1
                        elif flag == 0:
                            init_failure_1 = 0

            ihs = 6.28 * rng.random(num_robots)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, 0.032)
            pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

            for i in range(num_robots):
                pose.p = gymapi.Vec3(ixs[i], iys[i], 0.033)
                initial_positions[0][i] = pose.p.x  # Save initial position x of i'th robot
                initial_positions[1][i] = pose.p.y  # Save initial position y of i'th robot

                ihs_i = R.from_euler('zyx', [ihs[i], 0.0, 0.0])
                ihs_i = ihs_i.as_quat()
                pose.r.x = ihs_i[0]
                pose.r.y = ihs_i[1]
                pose.r.z = ihs_i[2]
                pose.r.w = ihs_i[3]

                # print("Loading asset '%s' from '%s', #%i" % (robot_asset_file, asset_root, i))
                robot_asset_file = individual[i].body
                robot_asset = gym.load_asset(
                    sim, asset_root, robot_asset_file, asset_options)

                # add robot
                robot_handle = gym.create_actor(env, robot_asset, pose, f"robot_{i}", 0, 0)
                robot_handles.append(robot_handle)
        else:
            distance = 0.5
            rows = 4

            # place robots
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

                # print("Loading asset '%s' from '%s', #%i" % (robot_asset_file, asset_root, i))
                robot_asset = gym.load_asset(
                    sim, asset_root, robot_asset_file, asset_options)

                # add robot
                robot_handle = gym.create_actor(env, robot_asset, pose, f"robot_{i}", 0, 0)
                robot_handles.append(robot_handle)
        robot_handles_list.append(np.array(robot_handles))

        # get joint limits and ranges for robot
        props = gym.get_actor_dof_properties(env, robot_handle)

        # Give a desired velocity to drive
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.05)
        props["damping"].fill(0.025)
        velocity_limits = props["velocity"]
        colors = []
        for i in range(num_robots):
            robot_handle = robot_handles[i]
            shape_props = gym.get_actor_rigid_shape_properties(env, robot_handle)
            shape_props[2].friction = 0
            shape_props[2].restitution = 0
            gym.set_actor_dof_properties(env, robot_handle, props)
            gym.set_actor_rigid_shape_properties(env, robot_handle, shape_props)
            if individual[i].phenotype["color"] != None:
                color = individual[i].phenotype["color"]
                colors.append(color)
                gym.set_rigid_body_color(env, robot_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

        positions = np.full([3, num_robots], 0.01)
        fitness_list.append(FitnessCalculator(num_robots, initial_positions, desired_movement,
                                              arena=arena,
                                              subgroup_ratio=subgroup_ratio,
                                              objectives=env_params['objectives']))
        sensor_list.append(Sensors(controller_types_list[i_env], arena=arena))

    def update_robot(env, controllers, robot_handles, states):
        for ii in range(len(states)):
            velocity_command = calc_vel_targets(controllers[ii], states[ii])
            gym.set_actor_dof_velocity_targets(env, robot_handles[ii], velocity_command)

    def get_pos_and_headings(env, robot_handles):
        headings = np.zeros((num_robots,))
        positions_x = np.zeros_like(headings)
        positions_y = np.zeros_like(headings)

        for i in range(num_robots):
            body_pose = gym.get_actor_rigid_body_states(env, robot_handles[i], gymapi.STATE_POS)["pose"][0]
            body_angle_mat = np.array(body_pose[1].tolist())
            r = R.from_quat(body_angle_mat)
            headings[i] = r.as_euler('zyx')[0]

            positions_x[i] = body_pose[0][0]
            positions_y[i] = body_pose[0][1]

        return headings, positions_x, positions_y

    t = 0
    timestep = 0  # Counter to save total time steps, required for final step of fitness value calculation
    fitness_mat = np.zeros((fitness_list[0].get_fitness_size(), len(individuals)))
    # Create viewer
    viewer = None
    plot = False
    record_video = env_params['record_video']
    save_full_fitness = env_params['save_full_fitness']
    if not headless:
        source_loc = fitness_list[0].source_pos
        x_dir = (ix - source_loc[0])
        y_dir = (iy - source_loc[1])
        vec = [x_dir/np.hypot(x_dir,y_dir), y_dir/np.hypot(x_dir, y_dir)]

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        ix_viewer = vec[0]*7.5 + ix
        iy_viewer = vec[1]*7.5 + iy
        z_viewer = -5
        if ix_viewer == ix and iy_viewer == iy:
            iy_viewer = iy - 7.5
            z_viewer = 5
        cam_pos = gymapi.Vec3(ix_viewer, iy_viewer, 5)
        cam_target = gymapi.Vec3(source_loc[0], source_loc[1], z_viewer)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        if len(individuals) == 1:
            plotter = swarm_plotter(arena, colors)  # Plotter init
            plot = True

            light_options = gymapi.AssetOptions()
            light_options.fix_base_link = True
            light_options.flip_visual_attachments = True
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(source_loc[0], source_loc[1], 4)
            ball_asset = gym.create_sphere(sim, 0.5, light_options)
            ahandle = gym.create_actor(env, ball_asset, pose, None, -1, -1)
            color = gymapi.Vec3(1.0, 1.0, 1.0)
            gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL, color)
            gym.set_light_parameters(sim, 0, gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(0.0, 0.0, 1.0))
            gym.set_light_parameters(sim, 1, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.1, 0.1, 0.1), gymapi.Vec3(0.0, 0.0, -1.0))

    # %% Simulate
    start = gym.get_sim_time(sim)
    frame = 0
    fitness_full = []
    while t <= life_timeout:
        t = gym.get_sim_time(sim)

        if (gym.get_sim_time(sim) - start) > 0.095:
            timestep += 1  # Time step counter
            for i_env in range(num_envs):
                env = env_list[i_env]
                robot_handles = robot_handles_list[i_env]
                controller_types = controller_types_list[i_env]
                controller = controller_list[i_env]
                # Update positions and headings of all robots
                headings, positions[0], positions[1] = get_pos_and_headings(env, robot_handles)
                sensor_list[i_env].calculate_states(positions, headings)
                states = sensor_list[i_env].get_current_state()
                update_robot(env, controller, robot_handles, states)

                fitness_mat[:, i_env] = fitness_list[i_env].obtain_fitnesses(positions, headings)/timestep
                if save_full_fitness:
                    fitness_full.append(copy.deepcopy(fitness_mat))
                    if not t < life_timeout:
                        np.save(f'./results/fitness_full.npy', np.array(fitness_full).squeeze())

            if plot:
                if record_video:
                    if (gym.get_sim_time(sim)%1) < 0.0005:
                        plotter.plot_swarm_quiver(positions, headings, frame)
                        gym.step_graphics(sim)
                        gym.draw_viewer(viewer, sim, False)
                        gym.write_viewer_image_to_file(viewer, f'./results/images/viewer/{frame}.png')
                        frame += 1
                else:
                    plotter.plot_swarm_quiver(positions, headings)
            start = gym.get_sim_time(sim)

        # Step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if not headless:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
    if not headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    return fitness_mat

def simulate_swarm_with_restart_population_split(life_timeout: float, individuals: list, swarm_size: int,
                                           headless: bool,
                                           objectives: list,
                                           splits: int,
                                           arena: str = "circle_30x30") -> np.ndarray:
    processes = []
    shared_memories = []
    results = []
    n= int(len(individuals) / splits)
    individuals_split = [individuals[i:i + n] for i in range(0, len(individuals), n)]
    for i, individuals_s in zip(range(splits), individuals_split):
        process, shared_memory, result = simulate_swarm_with_restart_population_start(
            life_timeout,individuals_s,swarm_size,headless,objectives,arena
        )
        processes.append(process)
        shared_memories.append(shared_memory)
        results.append(result)

    result_array = []
    for individuals_s, process, shared_memory, result in zip(individuals_split, processes, shared_memories, results):
        r = simulate_swarm_with_restart_population_end(process, shared_memory, result, len(individuals_s))
        result_array += r.tolist()

    return np.array(result_array)

def simulate_swarm_with_restart_population_start(life_timeout: float, individuals: list, swarm_size: int,
                                           headless: bool,
                                           objectives: list,
                                           arena: str = "circle_30x30"):
    """
    Obtains the results for simulate_swarm() with forced gpu memory clearance for restarts.

    :param individuals: robot phenotype for every member of the swarm
    :param life_timeout: how long should the robot live
    :param swarm_size: number of members inside the swarm
    :param headless: Start UI for debugging
    :param objectives: specify which components of the fitness function should be returned
    :param arena: Type of arena
    :return: fitness of the individual(s)
    """
    result = np.array([0.0] * len(individuals), dtype=float)
    shared_mem = shared_memory.SharedMemory(create=True, size=result.nbytes)
    process = Process(target=_inner_simulator_multiple_process_population,
                      args=(life_timeout, individuals, swarm_size, headless, objectives, arena, shared_mem.name))
    process.start()
    return process, shared_mem, result

def simulate_swarm_with_restart_population_end(process, shared_mem, result, pop_size: int) -> np.ndarray:
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f'Simulation for {process} exited with code {process.exitcode}')
    remote_result = np.ndarray((pop_size,), dtype=float, buffer=shared_mem.buf)
    result[:] = remote_result[:]
    shared_mem.close()
    shared_mem.unlink()
    return result


def _inner_simulator_multiple_process_population(life_timeout: float, individuals: list, swarm_size: int,
                                                 headless: bool,
                                                 objectives: list, arena,
                                                 shared_mem_name: AnyStr) -> int:
    existing_shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
    remote_result = np.ndarray((len(individuals),), dtype=float, buffer=existing_shared_mem.buf)
    try:
        fitness: np.ndarray = simulate_swarm_population(life_timeout, individuals, swarm_size, headless, objectives, arena)
        remote_result[:] = fitness
        existing_shared_mem.close()
        return 0
    except Exception as e:
        print(e)
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        remote_result[:] = -np.inf
        existing_shared_mem.close()
        return -1
