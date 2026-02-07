import argparse
import os
import time

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("pybullet is required. Install with: pip install pybullet") from exc

try:
    import smplx
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("smplx is required. Install with: pip install smplx") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("torch is required. Install with: pip install torch") from exc


STEP_RAD = 0.05

# SMPL-X body_pose has 21 joints (root rotation is global_orient).
SMPLX_BODY_JOINTS = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

CONTROL_JOINTS = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "spine1",
    "spine2",
    "spine3",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "neck",
    "head",
]


# Map SMPL-X joint -> robot joint (name fragment) + axis + scale.
# You will likely tweak these once you inspect robot joint names.
RETARGET_MAP = [
    ("left_hip", "left_hip", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_hip", "right_hip", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_knee", "left_knee", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_knee", "right_knee", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_ankle", "left_ankle", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_ankle", "right_ankle", np.array([1.0, 0.0, 0.0]), 1.0),
    ("spine1", "spine", np.array([1.0, 0.0, 0.0]), 0.5),
    ("spine2", "spine", np.array([1.0, 0.0, 0.0]), 0.5),
    ("spine3", "spine", np.array([1.0, 0.0, 0.0]), 0.5),
    ("left_shoulder", "left_shoulder", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_shoulder", "right_shoulder", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_elbow", "left_elbow", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_elbow", "right_elbow", np.array([1.0, 0.0, 0.0]), 1.0),
    ("neck", "neck", np.array([1.0, 0.0, 0.0]), 1.0),
    ("head", "head", np.array([1.0, 0.0, 0.0]), 1.0),
]


def resolve_model_folder(root: str) -> str:
    # Accept absolute path or relative to current working dir or this script location.
    candidates = [root]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "..", root))
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(candidate, "models")):
            return os.path.join(candidate, "models")
        if os.path.isdir(candidate):
            return candidate
    return os.path.abspath(root)


def resolve_model_file(root: str):
    # Accept direct .npz/.pkl path or search in folder.
    if os.path.isfile(root):
        return root
    for name in os.listdir(root):
        lower = name.lower()
        if lower.endswith(".npz") or lower.endswith(".pkl"):
            return os.path.join(root, name)
    return None


def create_smplx_model(model_folder: str, model_file: str | None):
    # If a direct model file is provided, pass it as the model_path.
    model_path = model_file or model_folder
    return smplx.create(
        model_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    )


def load_robot(urdf_path: str):
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    if os.path.isdir(urdf_dir):
        p.setAdditionalSearchPath(urdf_dir)
    data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(data_path)
    plane = p.loadURDF(os.path.join(data_path, "plane.urdf"))
    humanoid = p.loadURDF(urdf_path, useFixedBase=True, basePosition=[0, 0, 1.0])
    return plane, humanoid


def build_robot_joint_index(body_id: int):
    joint_index = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[1].decode("utf-8")
        joint_index[name] = i
    return joint_index


def pick_robot_joint(joint_index: dict, fragment: str):
    fragment = fragment.lower()
    for name, idx in joint_index.items():
        if fragment in name.lower():
            return idx
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    parser = argparse.ArgumentParser(description="SMPL-X to PyBullet humanoid retarget POC")
    parser.add_argument("--model-folder", default="assets/smplx", help="Folder with SMPL-X models/")
    parser.add_argument("--model-file", default=None, help="Optional SMPL-X model file (.npz/.pkl)")
    parser.add_argument(
        "--urdf",
        default=os.path.join(pybullet_data.getDataPath(), "humanoid/humanoid.urdf"),
        help="Path to humanoid URDF",
    )
    parser.add_argument("--print-robot-joints", action="store_true", help="List robot joint names and exit")
    args = parser.parse_args()

    model_folder = resolve_model_folder(args.model_folder)
    if not os.path.isdir(model_folder):
        raise SystemExit(f"Model folder not found: {model_folder}")

    model_file = args.model_file or resolve_model_file(model_folder)
    model = create_smplx_model(model_folder, model_file)
    body_pose = np.zeros((len(SMPLX_BODY_JOINTS), 3), dtype=np.float32)
    global_orient = np.zeros((1, 3), dtype=np.float32)
    joint_to_idx = {name: idx for idx, name in enumerate(SMPLX_BODY_JOINTS)}

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # Avoid flicker while loading.
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    _, robot_id = load_robot(args.urdf)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 1.0],
    )
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], lineWidth=2)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], lineWidth=2)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], lineWidth=2)
    visuals = p.getVisualShapeData(robot_id)
    print(f"Loaded robot id={robot_id}, joints={p.getNumJoints(robot_id)}, visuals={len(visuals)}")
    print(f"Robot base pose: {p.getBasePositionAndOrientation(robot_id)}")
    if len(visuals) == 0:
        print("Warning: robot has no visual shapes; check URDF mesh paths.")
    # Debug sphere to verify camera framing
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=[0, 0, 1.0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    joint_index = build_robot_joint_index(robot_id)

    if args.print_robot_joints:
        print("Robot joints:")
        for name in sorted(joint_index.keys()):
            print(name)
        return

    selected = 0
    last_text_id = None

    def update_hud():
        nonlocal last_text_id
        if last_text_id is not None:
            p.removeUserDebugItem(last_text_id)
        joint_name = CONTROL_JOINTS[selected]
        msg = f"Joint: {joint_name} | keys: arrows=Pitch/Yaw, Q/E=Roll, [ ]=select, R=reset, 0=reset all"
        last_text_id = p.addUserDebugText(msg, [0, 0, 1.8], textSize=1.2)

    update_hud()

    while True:
        keys = p.getKeyboardEvents()
        if keys:
            if ord("[") in keys and keys[ord("[")] & p.KEY_WAS_TRIGGERED:
                selected = (selected - 1) % len(CONTROL_JOINTS)
                update_hud()
            if ord("]") in keys and keys[ord("]")] & p.KEY_WAS_TRIGGERED:
                selected = (selected + 1) % len(CONTROL_JOINTS)
                update_hud()
            if ord("0") in keys and keys[ord("0")] & p.KEY_WAS_TRIGGERED:
                body_pose[:] = 0
            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                body_pose[joint_to_idx[CONTROL_JOINTS[selected]]] = 0

            delta = np.zeros(3, dtype=np.float32)
            if p.B3G_UP_ARROW in keys:
                delta[0] += STEP_RAD
            if p.B3G_DOWN_ARROW in keys:
                delta[0] -= STEP_RAD
            if p.B3G_LEFT_ARROW in keys:
                delta[1] += STEP_RAD
            if p.B3G_RIGHT_ARROW in keys:
                delta[1] -= STEP_RAD
            if ord("q") in keys:
                delta[2] += STEP_RAD
            if ord("e") in keys:
                delta[2] -= STEP_RAD
            if np.any(delta):
                idx = joint_to_idx[CONTROL_JOINTS[selected]]
                body_pose[idx] += delta
                body_pose[idx] = np.array(
                    [clamp(v, -2.0, 2.0) for v in body_pose[idx]], dtype=np.float32
                )

        # Forward SMPL-X (not used for mesh rendering here, but validates the model path)
        # smplx expects torch tensors
        _ = model(
            global_orient=torch.from_numpy(global_orient).float(),
            body_pose=torch.from_numpy(body_pose[None, ...]).float(),
        )

        # Retarget to robot joints
        for smpl_joint, robot_fragment, axis, scale in RETARGET_MAP:
            if smpl_joint not in joint_to_idx:
                continue
            smpl_idx = joint_to_idx[smpl_joint]
            aa = body_pose[smpl_idx]
            angle = float(np.dot(axis, aa)) * scale
            robot_idx = pick_robot_joint(joint_index, robot_fragment)
            if robot_idx is None:
                continue
            p.setJointMotorControl2(
                robot_id,
                robot_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=150,
            )

        p.stepSimulation()
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
