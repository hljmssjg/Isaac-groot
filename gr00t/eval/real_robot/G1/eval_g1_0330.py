"""
G1_0330 Real-Robot GR00T Policy Evaluation Script

This script runs closed-loop policy evaluation on the Unitree G1 robot
using the GR00T Policy API, interfacing directly with the G1 hardware
via the unitree SDK (same stack as xr_teleoperate used for data collection).

Architecture:
    [ImageServer (ZMQ PUB)] --> [This Script] --ZMQ--> [GR00T Inference Server (GPU)]
                                     |
    [G1 Robot (DDS lowstate)] <------+------> [G1 Robot (DDS motor cmd)]

Data collection was done with xr_teleoperate, which uses:
    - ImageServer: head(opencv) + wrist(realsense D405) + extra(realsense L515)
      concatenated into a single JPEG over ZMQ PUB
    - G1_29_ArmController: unitree DDS for 14 DOF arm state/control
    - Dex3_1_Controller: unitree DDS for 14 DOF hand state/control

The model expects 28 DOF state in this order:
    left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7)

Prerequisites:
    1. ImageServer running on the robot (image_server.py with camera_config.json)
    2. GR00T inference server running on GPU (run_gr00t_server.py)
    3. G1 robot powered on with DDS accessible

Usage:
    python gr00t/eval/real_robot/G1/eval_g1_0330.py \
        --policy_host <GPU_SERVER_IP> \
        --policy_port 5555 \
        --action_horizon 16 \
        --lang_instruction "<task description>" \
        --camera_host <ROBOT_IP> \
        --camera_port 5555
"""

import argparse
import logging
import os
import struct
import sys
import time
from typing import Any, Dict, List

# Ensure xr_teleoperate and unitree_IL_lerobot are importable
sys.path.insert(0, os.path.expanduser("~/xr_teleoperate"))
sys.path.insert(0, os.path.expanduser("~/unitree_IL_lerobot"))

import cv2
import numpy as np
import zmq

from gr00t.policy.server_client import PolicyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Image Client - receives concatenated JPEG from xr_teleoperate ImageServer
# =============================================================================

class ImageReceiver:
    """
    Receives concatenated camera frames from xr_teleoperate's ImageServer.

    The ImageServer concatenates head + wrist + extra images horizontally
    into a single JPEG and publishes via ZMQ PUB socket.

    This receiver decodes and splits them back into individual camera images,
    mapped to the GR00T modality config keys (cam_head, cam_left_wrist, cam_extra).
    """

    def __init__(
        self,
        server_address: str,
        port: int,
        head_width: int = 640,
        wrist_width: int = 640,
        extra_width: int = 960,
    ):
        self.head_width = head_width
        self.wrist_width = wrist_width
        self.extra_width = extra_width

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{server_address}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Don't block if no image available yet
        self.socket.setsockopt(zmq.RCVTIMEO, 100)

        logger.info(
            f"ImageReceiver connected to {server_address}:{port}, "
            f"expecting widths: head={head_width}, wrist={wrist_width}, extra={extra_width}"
        )

    def receive(self) -> Dict[str, np.ndarray] | None:
        """
        Receive and split concatenated image into individual camera images.

        Returns:
            Dict with keys "cam_head", "cam_left_wrist", "cam_extra" -> np.ndarray (H, W, 3) BGR
            or None if no image available.
        """
        try:
            message = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None

        np_img = np.frombuffer(message, dtype=np.uint8)
        full_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if full_image is None:
            logger.warning("Failed to decode image from ImageServer")
            return None

        # Split the horizontally concatenated image
        cursor = 0
        images = {}

        # Head camera
        images["cam_head"] = full_image[:, cursor:cursor + self.head_width]
        cursor += self.head_width

        # Wrist camera (may be resized by ImageServer to match head height)
        if self.wrist_width > 0 and cursor + self.wrist_width <= full_image.shape[1]:
            images["cam_left_wrist"] = full_image[:, cursor:cursor + self.wrist_width]
            cursor += self.wrist_width

        # Extra camera (may be resized by ImageServer to match head height)
        remaining_width = full_image.shape[1] - cursor
        if remaining_width > 0:
            images["cam_extra"] = full_image[:, cursor:cursor + remaining_width]

        return images

    def close(self):
        self.socket.close()
        self.context.term()


# =============================================================================
# Robot State/Control - direct DDS interface via unitree SDK
# =============================================================================

class G1GravityCompensator:
    """
    Computes gravity compensation feedforward torque for G1_29 arm joints
    using Pinocchio RNEA, matching xr_teleoperate's data collection behavior.

    Builds a reduced Pinocchio model (14-DOF arms only) by locking all
    non-arm joints (legs, waist, hands) at zero configuration.
    """

    # Joints to lock (everything except the 14 arm joints)
    _JOINTS_TO_LOCK = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]

    def __init__(self, urdf_path: str, mesh_dir: str):
        import pinocchio as pin

        self._pin = pin
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self._reduced = robot.buildReducedRobot(
            list_of_joints_to_lock=self._JOINTS_TO_LOCK,
            reference_configuration=np.array([0.0] * robot.model.nq),
        )
        self._nv = self._reduced.model.nv
        logger.info(f"GravityCompensator initialized: {self._nv} DOF reduced model")

    def compute(self, arm_q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torque for 14-DOF arm joint targets.

        Uses RNEA with zero velocity and zero acceleration, yielding pure
        gravity + static friction terms: tau = g(q).
        """
        return self._pin.rnea(
            self._reduced.model,
            self._reduced.data,
            arm_q,
            np.zeros(self._nv),
            np.zeros(self._nv),
        )


class G1RobotInterface:
    """
    Direct interface to G1 robot hardware via unitree SDK (DDS).
    Same joint ordering as xr_teleoperate used during data collection.

    Arm joints (G1_29_JointArmIndex):
        left: ShoulderPitch(15), ShoulderRoll(16), ShoulderYaw(17),
              Elbow(18), WristRoll(19), WristPitch(20), WristYaw(21)
        right: ShoulderPitch(22), ShoulderRoll(23), ShoulderYaw(24),
               Elbow(25), WristRoll(26), WristPitch(27), WristYaw(28)

    Hand joints (Dex3_1 JointIndex):
        left:  Thumb0(0), Thumb1(1), Thumb2(2), Middle0(3), Middle1(4), Index0(5), Index1(6)
        right: Thumb0(0), Thumb1(1), Thumb2(2), Index0(3), Index1(4), Middle0(5), Middle1(6)
    """

    def __init__(self, simulation_mode: bool = False,
                 urdf_path: str = None, mesh_dir: str = None):
        from multiprocessing import Array, Lock
        from teleop.robot_control.robot_arm import (
            G1_29_ArmController,
            G1_29_JointIndex,
            G1_29_Num_Motors,
        )
        from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
            Dex3_1_Controller,
            Dex3_Num_Motors,
        )

        logger.info("Initializing G1 robot interface...")

        # Build sitting-pose preset for ALL body joints
        preset_body_q = np.zeros(G1_29_Num_Motors)
        _sitting_pose = {
            "kLeftHipPitch": -1.3,  "kLeftKnee": 1.3,
            "kRightHipPitch": -1.3, "kRightKnee": 1.3,
        }
        for jid in G1_29_JointIndex:
            if jid.name in _sitting_pose:
                preset_body_q[jid] = _sitting_pose[jid.name]

        # Initialize arm controller
        self.arm_ctrl = G1_29_ArmController(
            simulation_mode=simulation_mode,
            preset_body_q=preset_body_q,
        )

        # Initialize gravity compensator
        if urdf_path and mesh_dir:
            self.gravity_comp = G1GravityCompensator(urdf_path, mesh_dir)
        else:
            logger.warning("No URDF provided — gravity compensation DISABLED (tauff=0)")
            self.gravity_comp = None

        # Initialize hand controller (unitree_IL_lerobot's Dex3_1_Controller).
        # Subprocess continuously sends shared-memory values at 100Hz.
        self._hand_left_in = Array("d", [0.0] * Dex3_Num_Motors, lock=True)
        self._hand_right_in = Array("d", [0.0] * Dex3_Num_Motors, lock=True)
        self._hand_data_lock = Lock()
        self._hand_state_array = Array("d", Dex3_Num_Motors * 2, lock=False)
        self._hand_action_array = Array("d", Dex3_Num_Motors * 2, lock=False)

        self.hand_ctrl = Dex3_1_Controller(
            self._hand_left_in,
            self._hand_right_in,
            self._hand_data_lock,
            self._hand_state_array,
            self._hand_action_array,
            simulation_mode=simulation_mode,
        )
        logger.info("G1 robot interface initialized")

    def _send_hand_targets(self, left_q: np.ndarray, right_q: np.ndarray):
        """Write hand joint targets into shared memory; subprocess sends at 100Hz."""
        with self._hand_left_in.get_lock():
            self._hand_left_in[:] = left_q.tolist()
        with self._hand_right_in.get_lock():
            self._hand_right_in[:] = right_q.tolist()

    def init_to_start_pose(self, init_state_28: np.ndarray, timeout: float = 5.0,
                           tolerance: float = 0.05):
        """
        Move arm and hand joints to the episode starting pose.
        Blocks until arm joints converge or timeout.
        """
        init_arm = init_state_28[:14]
        init_left_hand = init_state_28[14:21]
        init_right_hand = init_state_28[21:28]

        # Send arm target with gravity compensation
        if self.gravity_comp is not None:
            tauff = self.gravity_comp.compute(init_arm)
        else:
            tauff = np.zeros(14)
        self.arm_ctrl.ctrl_dual_arm(init_arm, tauff)

        # Send hand targets via shared memory
        self._send_hand_targets(init_left_hand, init_right_hand)

        # Wait for arm convergence
        logger.info("Waiting for arm to reach start pose...")
        t_start = time.monotonic()
        while time.monotonic() - t_start < timeout:
            current_arm = self.arm_ctrl.get_current_dual_arm_q()
            if np.all(np.abs(current_arm - init_arm) < tolerance):
                logger.info("Arm reached start pose.")
                return
            time.sleep(0.05)
        logger.warning(f"Arm convergence timeout ({timeout}s). Proceeding anyway.")

    def restore_init_pose(self, init_state_28: np.ndarray, timeout: float = 5.0,
                          tolerance: float = 0.05):
        """Restore robot to initial pose after evaluation ends."""
        self.init_to_start_pose(init_state_28, timeout=timeout, tolerance=tolerance)
        logger.info("Holding init pose for 3 seconds...")
        time.sleep(3.0)

    def get_state(self) -> np.ndarray:
        """
        Get current 28-DOF state: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7).
        """
        arm_q = self.arm_ctrl.get_current_dual_arm_q()
        with self._hand_data_lock:
            hand_state = np.array(self._hand_state_array[:])
        return np.concatenate([arm_q, hand_state]).astype(np.float32)

    def send_action(self, action: np.ndarray):
        """
        Send 28-DOF action: left_arm(7) + right_arm(7) + left_hand(7) + right_hand(7).
        """
        arm_action = action[:14]
        left_hand_action = action[14:21]
        right_hand_action = action[21:28]

        if self.gravity_comp is not None:
            tauff = self.gravity_comp.compute(arm_action)
        else:
            tauff = np.zeros(14)
        self.arm_ctrl.ctrl_dual_arm(arm_action, tauff)
        self._send_hand_targets(left_hand_action, right_hand_action)


# =============================================================================
# GR00T Policy Adapter
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.
    GR00T Policy Server expects obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class G1_0330Adapter:
    """
    Adapter between G1 robot observations and GR00T VLA format.

    State format (28-DOF flat array from xr_teleoperate):
        [0:7]   left_arm   - ShoulderPitch/Roll/Yaw, Elbow, WristRoll/Pitch/Yaw
        [7:14]  right_arm  - same order
        [14:21] left_hand  - Thumb0/1/2, Middle0/1, Index0/1
        [21:28] right_hand - Thumb0/1/2, Index0/1, Middle0/1

    Note: The GR00T inference server internally calls `unapply_action` which
    handles both unnormalization and relative-to-absolute conversion. The action
    chunk returned by PolicyClient already contains absolute joint angles.
    """

    # Joint group slices matching the flat 28-DOF state vector and modality.json
    JOINT_GROUPS = {
        "left_arm":   slice(0, 7),
        "right_arm":  slice(7, 14),
        "left_hand":  slice(14, 21),
        "right_hand": slice(21, 28),
    }

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

    def obs_to_policy_inputs(
        self,
        state_28: np.ndarray,
        images: Dict[str, np.ndarray],
        lang_instruction: str,
    ) -> Dict:
        """
        Convert robot observations into GR00T VLA input format.

        Args:
            state_28: 28-DOF state vector [left_arm, right_arm, left_hand, right_hand]
            images: Camera images dict with keys "cam_head", "cam_left_wrist", "cam_extra"
            lang_instruction: Language task description
        """
        model_obs = {}

        # (1) Cameras - convert BGR (from OpenCV) to RGB (model trained on RGB)
        model_obs["video"] = {
            k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
            for k, v in images.items()
            if k in ("cam_head", "cam_left_wrist", "cam_extra")
        }

        # (2) Joint states - split into per-group arrays
        model_obs["state"] = {
            name: state_28[slc].copy()
            for name, slc in self.JOINT_GROUPS.items()
        }

        # (3) Language instruction
        model_obs["language"] = {
            "annotation.human.task_description": lang_instruction,
        }

        # (4) Add (B=1, T=1) dims
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int) -> np.ndarray:
        """
        Decode a single timestep from the action chunk into a flat 28-DOF array.

        The action chunk from PolicyClient already contains absolute joint angles.
        """
        parts = []
        for name in self.JOINT_GROUPS:
            parts.append(chunk[name][0][t])  # (D,) - remove batch dim, select timestep
        return np.concatenate(parts)  # (28,)

    def get_action(
        self,
        state_28: np.ndarray,
        images: Dict[str, np.ndarray],
        lang_instruction: str,
    ) -> List[np.ndarray]:
        """
        Full pipeline: observation -> model inference -> decoded action sequence.

        Returns:
            List of 28-DOF absolute joint angle arrays, one per action horizon step.
        """
        model_input = self.obs_to_policy_inputs(state_28, images, lang_instruction)
        action_chunk, info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B, T, D) -> T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="G1_0330 Real-Robot GR00T Policy Evaluation"
    )
    parser.add_argument("--policy_host", type=str, default="localhost",
                        help="GR00T inference server host IP")
    parser.add_argument("--policy_port", type=int, default=5555,
                        help="GR00T inference server port")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Number of action steps to execute per inference call")
    parser.add_argument("--lang_instruction", type=str, default="Pick up the object.",
                        help="Language task instruction for the policy")
    parser.add_argument("--control_frequency", type=float, default=30.0,
                        help="Control loop frequency in Hz (should match data collection FPS)")
    parser.add_argument("--camera_host", type=str, default="192.168.123.164",
                        help="ImageServer host (robot IP)")
    parser.add_argument("--camera_port", type=int, default=5555,
                        help="ImageServer port")
    parser.add_argument("--head_width", type=int, default=640,
                        help="Head camera image width in concatenated frame")
    parser.add_argument("--wrist_width", type=int, default=640,
                        help="Wrist camera image width in concatenated frame")
    parser.add_argument("--extra_width", type=int, default=960,
                        help="Extra camera image width in concatenated frame")
    parser.add_argument("--sim", action="store_true", default=False,
                        help="Use simulation mode (no real robot)")
    parser.add_argument("--init_state", type=str, default=None,
                        help="Path to 28-DOF init state .txt file. If omitted, --dataset_path must be given "
                             "and episode 0 frame 0 is used automatically.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to LeRobot dataset (e.g. cache/Jiangeng/G1_0330). When provided, "
                             "init state is read from episode_000000.parquet frame 0.")
    parser.add_argument("--init_episode", type=int, default=0,
                        help="Episode index to read init state from when --dataset_path is used.")
    parser.add_argument("--urdf", type=str,
                        default=os.path.expanduser("~/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"),
                        help="Path to G1 URDF for gravity compensation")
    parser.add_argument("--mesh_dir", type=str,
                        default=os.path.expanduser("~/xr_teleoperate/assets/g1/"),
                        help="Path to URDF mesh directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load 28-DOF init state
    #    Priority: --init_state file > --dataset_path episode 0 frame 0
    # -------------------------------------------------------------------------
    if args.init_state is not None:
        init_state_28 = np.loadtxt(args.init_state).astype(np.float32)
        logger.info(f"Loaded init state from {args.init_state}")
    elif args.dataset_path is not None:
        import glob
        import pandas as pd
        pattern = os.path.join(args.dataset_path, "data", "**",
                               f"episode_{args.init_episode:06d}.parquet")
        matches = sorted(glob.glob(pattern, recursive=True))
        assert matches, f"No episode parquet matched: {pattern}"
        df = pd.read_parquet(matches[0])
        init_state_28 = np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
        logger.info(
            f"Loaded init state from {matches[0]} "
            f"(episode {args.init_episode} frame 0)"
        )
    else:
        raise SystemExit("Must provide either --init_state or --dataset_path")

    assert init_state_28.shape == (28,), (
        f"init_state must be 28-DOF, got shape {init_state_28.shape}"
    )

    # -------------------------------------------------------------------------
    # 2. Initialize Camera Receiver
    # -------------------------------------------------------------------------
    image_receiver = ImageReceiver(
        server_address=args.camera_host,
        port=args.camera_port,
        head_width=args.head_width,
        wrist_width=args.wrist_width,
        extra_width=args.extra_width,
    )

    # -------------------------------------------------------------------------
    # 3. Initialize Robot Interface
    #    ArmController immediately moves body to sitting pose (legs/waist)
    # -------------------------------------------------------------------------
    robot = G1RobotInterface(
        simulation_mode=args.sim,
        urdf_path=args.urdf,
        mesh_dir=args.mesh_dir,
    )

    # -------------------------------------------------------------------------
    # 4. Initialize Policy Client and Adapter
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=args.policy_host, port=args.policy_port)
    adapter = G1_0330Adapter(policy_client)
    logger.info(f"Policy client connected to {args.policy_host}:{args.policy_port}")
    logger.info(f'Language instruction: "{args.lang_instruction}"')

    # -------------------------------------------------------------------------
    # 5. Wait for user confirmation, then move to episode start pose
    # -------------------------------------------------------------------------
    logger.info("Robot is moving to sitting pose (legs/waist)...")
    logger.info(f"Init arm state:  {init_state_28[:14]}")
    logger.info(f"Init hand state: {init_state_28[14:]}")
    user_input = input("Enter 's' to move arm+hand to start pose and begin evaluation: ")
    if user_input.strip().lower() != "s":
        logger.info("Aborted by user.")
        return

    robot.init_to_start_pose(init_state_28)

    # -------------------------------------------------------------------------
    # 6. Wait for camera images
    # -------------------------------------------------------------------------
    logger.info("Waiting for camera images...")
    latest_images = None
    while latest_images is None:
        latest_images = image_receiver.receive()
        time.sleep(0.01)
    logger.info(f"Camera ready. Image keys: {list(latest_images.keys())}")

    state = robot.get_state()
    logger.info(f"Robot state ready. Shape: {state.shape}")
    logger.info("Starting control loop.")

    # -------------------------------------------------------------------------
    # 7. Main Control Loop
    # -------------------------------------------------------------------------
    control_dt = 1.0 / args.control_frequency
    iteration = 0

    try:
        while True:
            loop_start = time.monotonic()

            # --- Poll latest images (non-blocking, keep last if none) ---
            new_images = image_receiver.receive()
            if new_images is not None:
                latest_images = new_images

            # --- Read current robot state ---
            state = robot.get_state()

            # --- Run inference ---
            t_infer = time.monotonic()
            actions = adapter.get_action(
                state_28=state,
                images=latest_images,
                lang_instruction=args.lang_instruction,
            )
            infer_time = time.monotonic() - t_infer

            if iteration == 0:
                logger.info(f"First inference took {infer_time:.3f}s, "
                            f"action horizon: {len(actions)}")

            # --- Execute action chunk ---
            for i, action in enumerate(actions[:args.action_horizon]):
                t_action = time.monotonic()
                robot.send_action(action)

                # Maintain control frequency
                elapsed = time.monotonic() - t_action
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            iteration += 1

            if iteration % 50 == 0:
                total_time = time.monotonic() - loop_start
                logger.info(
                    f"Iter {iteration}: total={total_time:.3f}s, "
                    f"infer={infer_time:.3f}s, "
                    f"state[:4]={state[:4]}"
                )

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")

    finally:
        logger.info("Restoring to init pose...")
        robot.restore_init_pose(init_state_28)
        image_receiver.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
