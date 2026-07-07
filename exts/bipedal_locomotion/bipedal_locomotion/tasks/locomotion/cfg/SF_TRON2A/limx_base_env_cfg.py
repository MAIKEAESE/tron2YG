import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from bipedal_locomotion.tasks.locomotion import mdp

##################
# Scene Definition
##################


@configclass
class SF_TRON2A_SceneCfg(InteractiveSceneCfg):
    """Configuration for the SF_TRON2A scene"""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # bipedal robot
    robot: ArticulationCfg = MISSING

    # height sensors
    height_scanner: RayCasterCfg = MISSING

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(0.8, 1.0),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
            swing_height=(0.10, 0.20),
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.10,
        rel_heading_envs=1.0,
        #debug_vis=True,
        debug_vis=False,
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


joint_order_name = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "ankle_pitch_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
    "ankle_pitch_R_Joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_order_name,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=1.0,
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=1.0,
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.0,
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_order_name, preserve_order=True)},
            scale=0.05,
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # heights scan
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.8},
            clip=(-1.0, 1.0),
            scale=5.0,
        )

        # Privileged observation
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, scale=0.05)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, scale=0.0025)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*")},
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*")},
            scale=0.01,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    prepare_quantity_for_tron1_piper = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_radius": 0.074,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_[LR]_Link"]),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 48,
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # Reward terms: ---Task
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.30)},
    )
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.30)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    base_height_exp = RewTerm(
        func=mdp.base_height_exp,
        weight=0.6,
        params={"std": math.sqrt(0.005)},
    )
    keep_balance = RewTerm(func=mdp.stay_alive, weight=1.0)

    # Reward terms: ---Gait
    gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=0.5,
        params={
            "tracking_contacts_shaped_force": 1.0,
            "tracking_contacts_shaped_vel": 1.0,
            "tracking_contacts_shaped_height": -0.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "gait_height_sigma": 0.005,
            "touch_down_vel": 0.0,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "use_reference_motion": True,
        },
    )

    # Reward terms: ---Base regulation
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    body_lin_acc = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Reward terms: ---Action regulation
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # same
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.0004)

    # Reward terms: ---Contact regulation
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!ankle_pitch_).*"), "threshold": 10.0},
    )

    # Reward terms: ---Body control
    feet_distance = RewTerm(
        func=mdp.distance_aligned,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "min_dist": 0.20,
            "max_dist": 0.60,
            "desired_dist": 0.40,
            "std": math.sqrt(0.01),
            "command_name": "base_velocity",
            "vy_max": 0.8,
            "decay_power": 1.0,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
        },
    )
    feet_ang_slide_penalty = RewTerm(
        func=mdp.feet_angle_slide,
        weight=-0.10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
        },
    )

    # Reward terms: ---Joint regulation
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-4e-7)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-7)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.025,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 0.92},
    )
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),},
    )
    dof_power_l1 = RewTerm(
        func=mdp.weighted_joint_power_l1,
        weight=-2.5e-7,
        params={
            "power_weight": {
                "proximal_pitch_L_Joint": 0.01,
                "proximal_roll_L_Joint": 0.1,
                "proximal_yaw_L_Joint": 0.1,
                "knee_L_Joint": 0.01,
                "ankle_pitch_L_Joint": 0.1,
                "proximal_pitch_R_Joint": 0.01,
                "proximal_roll_R_Joint": 0.1,
                "proximal_yaw_R_Joint": 0.1,
                "knee_R_Joint": 0.01,
                "ankle_pitch_R_Joint": 0.1,
            }
        },
    )
    joint_deviation_l1 = RewTerm(
        func=mdp.weighted_joint_deviation_l1,
        weight=-1.0,
        params={
            "deviation_weight": {
                "proximal_pitch_L_Joint": 0.005,
                "proximal_roll_L_Joint": 0.005,
                "proximal_yaw_L_Joint": 0.005,
                "knee_L_Joint": 0.005,
                "ankle_pitch_L_Joint": 0.005,
                "proximal_pitch_R_Joint": 0.005,
                "proximal_roll_R_Joint": 0.005,
                "proximal_yaw_R_Joint": 0.005,
                "knee_R_Joint": 0.005,
                "ankle_pitch_R_Joint": 0.005,
            }
        },
    )
    knee_joint_orientation = RewTerm(
        func=mdp.joint_orientation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["knee_.*"])},
    )
    foot_joint_orientation = RewTerm(
        func=mdp.joint_orientation_l1_symmetric,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ankle_pitch.*"])},
    )
    orientation_exp_knee = RewTerm(
        func=mdp.body_orientation_yaw_exp,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="knee_.*"), "target_yaw": math.pi},
    )
    torques_smoothness_penalty = RewTerm(func=mdp.torques_smoothness_penalty, weight=-1.5e-7)
    foot_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[RL]_Link"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_Link"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(80.0)})
    action_out_of_limits = DoneTerm(
        func=mdp.action_out_of_limits,
        params={
            "threshold": 100.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    pass


########################
# Environment definition
########################


@configclass
class SF_TRON2A_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SF_TRON2A environment"""

    # Scene settings
    scene: SF_TRON2A_SceneCfg = SF_TRON2A_SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
