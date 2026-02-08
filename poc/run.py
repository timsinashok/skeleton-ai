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


STEP_RAD = 0.12

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


# Map SMPL-X joint -> robot joint name + axis + scale.
RETARGET_MAP = [
    ("left_hip", "left_hip", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_hip", "right_hip", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_knee", "left_knee", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_knee", "right_knee", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_ankle", "left_ankle", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_ankle", "right_ankle", np.array([1.0, 0.0, 0.0]), 1.0),
    ("spine1", "chest", np.array([1.0, 0.0, 0.0]), 0.5),
    ("spine2", "chest", np.array([1.0, 0.0, 0.0]), 0.5),
    ("spine3", "chest", np.array([1.0, 0.0, 0.0]), 0.5),
    ("left_shoulder", "left_shoulder", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_shoulder", "right_shoulder", np.array([1.0, 0.0, 0.0]), 1.0),
    ("left_elbow", "left_elbow", np.array([1.0, 0.0, 0.0]), 1.0),
    ("right_elbow", "right_elbow", np.array([1.0, 0.0, 0.0]), 1.0),
    ("neck", "neck", np.array([1.0, 0.0, 0.0]), 1.0),
    ("head", "neck", np.array([1.0, 0.0, 0.0]), 0.3),
]

# Rest pose offsets.
ROBOT_STAND_SCALAR = {
    "left_knee": -0.55,
    "right_knee": -0.55,
    "left_elbow": 0.55,
    "right_elbow": 0.55,
}

ROBOT_STAND_EULER = {
    "left_hip": np.array([0.15, 0.00, -0.05], dtype=np.float32),
    "right_hip": np.array([0.15, 0.00, 0.05], dtype=np.float32),
    "left_ankle": np.array([-0.10, 0.00, 0.00], dtype=np.float32),
    "right_ankle": np.array([-0.10, 0.00, 0.00], dtype=np.float32),
    "left_shoulder": np.array([0.10, 0.00, 0.00], dtype=np.float32),
    "right_shoulder": np.array([0.10, 0.00, 0.00], dtype=np.float32),
    "chest": np.array([0.00, 0.00, 0.00], dtype=np.float32),
    "neck": np.array([0.00, 0.00, 0.00], dtype=np.float32),
}


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


def resolve_robot_joint(joint_index: dict, key: str):
    if key in joint_index:
        return joint_index[key]
    return pick_robot_joint(joint_index, key)


def build_joint_limits(body_id: int):
    limits = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        lower = float(info[8])
        upper = float(info[9])
        limits[i] = (lower, upper)
    return limits


def build_joint_types(body_id: int):
    types = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        types[i] = int(info[2])
    return types


def build_robot_tree(body_id: int):
    edges = []
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        parent = info[16]
        edges.append((parent, i))
    return edges


def get_link_world_pos(body_id: int, link_idx: int):
    if link_idx == -1:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return pos
    state = p.getLinkState(body_id, link_idx, computeForwardKinematics=True)
    return state[0]


def pick_robot_joint(joint_index: dict, fragment: str):
    fragment = fragment.lower()
    for name, idx in joint_index.items():
        if fragment in name.lower():
            return idx
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_to_joint_limits(value: float, lower: float, upper: float) -> float:
    # Some joints expose invalid/no limits; only clamp when limits are valid.
    if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
        return float(clamp(value, lower, upper))
    return float(value)


def apply_joint_target(body_id: int, joint_idx: int, joint_type: int, angle_vec: np.ndarray, scalar: float):
    # 2 = JOINT_SPHERICAL, 0 = JOINT_REVOLUTE
    if joint_type == p.JOINT_SPHERICAL:
        quat = p.getQuaternionFromEuler(angle_vec.tolist())
        p.resetJointStateMultiDof(
            body_id,
            joint_idx,
            targetValue=quat,
            targetVelocity=[0.0, 0.0, 0.0],
        )
    elif joint_type == p.JOINT_REVOLUTE:
        p.resetJointState(body_id, joint_idx, targetValue=scalar, targetVelocity=0.0)


def key_is_down(keys: dict, key_code: int) -> bool:
    return bool(keys.get(key_code, 0) & p.KEY_IS_DOWN)


def key_triggered(keys: dict, key_code: int) -> bool:
    return bool(keys.get(key_code, 0) & p.KEY_WAS_TRIGGERED)


def main():
    parser = argparse.ArgumentParser(description="SMPL-X to PyBullet humanoid retarget POC")
    parser.add_argument("--model-folder", default="assets/smplx", help="Folder with SMPL-X models/")
    parser.add_argument("--model-file", default=None, help="Optional SMPL-X model file (.npz/.pkl)")
    parser.add_argument(
        "--urdf",
        default=os.path.join(pybullet_data.getDataPath(), "humanoid/humanoid.urdf"),
        help="Path to humanoid URDF",
    )
    parser.add_argument("--wireframe", action="store_true", help="Enable wireframe rendering")
    parser.add_argument("--debug-keys", action="store_true", help="Print key event codes seen by PyBullet")
    parser.add_argument("--self-test", action="store_true", help="Drive joints with a sine wave (no keyboard needed)")
    parser.add_argument(
        "--gui-options",
        default="",
        help="Optional PyBullet GUI options, e.g. '--opengl2'",
    )
    parser.add_argument("--print-robot-joints", action="store_true", help="List robot joint names and exit")
    args = parser.parse_args()

    model_folder = resolve_model_folder(args.model_folder)
    if not os.path.isdir(model_folder):
        raise SystemExit(f"Model folder not found: {model_folder}")

    if args.model_file is not None and args.model_file.startswith("--"):
        raise SystemExit("Invalid --model-file value. Provide an absolute path to .npz/.pkl.")

    model_file = args.model_file or resolve_model_file(model_folder)
    if model_file is not None and not os.path.isfile(model_file):
        raise SystemExit(f"Model file not found: {model_file}")
    model = create_smplx_model(model_folder, model_file)
    body_pose = np.zeros((len(SMPLX_BODY_JOINTS), 3), dtype=np.float32)
    global_orient = np.zeros((1, 3), dtype=np.float32)
    joint_to_idx = {name: idx for idx, name in enumerate(SMPLX_BODY_JOINTS)}

    if args.gui_options.strip():
        p.connect(p.GUI, options=args.gui_options)
    else:
        p.connect(p.GUI)
    # Kinematic POC: disable gravity to avoid physics instability.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    if args.wireframe:
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
    _, robot_id = load_robot(args.urdf)
    upright_base_orn = p.getQuaternionFromEuler([np.pi / 2.0, 0.0, 0.0])
    upright_base_pos = [0.0, 0.0, 3.6]
    p.resetBasePositionAndOrientation(robot_id, upright_base_pos, upright_base_orn)
    p.resetDebugVisualizerCamera(
        cameraDistance=6.5,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1.8],
    )
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], lineWidth=2)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], lineWidth=2)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], lineWidth=2)
    visuals = p.getVisualShapeData(robot_id)
    print(f"Loaded robot id={robot_id}, joints={p.getNumJoints(robot_id)}, visuals={len(visuals)}")
    print(f"Robot base pose: {p.getBasePositionAndOrientation(robot_id)}")
    if len(visuals) == 0:
        print("Warning: robot has no visual shapes; check URDF mesh paths.")
    # Force bright colors so the robot is visible.
    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, link_index, rgbaColor=[1, 0.2, 0.2, 1])
    # Debug sphere to verify camera framing
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=[0, 0, 1.0])
    joint_index = build_robot_joint_index(robot_id)
    joint_limits = build_joint_limits(robot_id)
    joint_types = build_joint_types(robot_id)
    robot_joint_targets = {}
    for _, robot_name, _, _ in RETARGET_MAP:
        if robot_name in joint_index:
            robot_joint_targets[robot_name] = joint_index[robot_name]
        else:
            idx = resolve_robot_joint(joint_index, robot_name)
            if idx is not None:
                robot_joint_targets[robot_name] = idx

    # Apply initial standing offsets for both spherical and revolute joints.
    for joint_name, offset in ROBOT_STAND_SCALAR.items():
        idx = joint_index.get(joint_name)
        if idx is not None:
            lower, upper = joint_limits.get(idx, (-np.inf, np.inf))
            offset = clamp_to_joint_limits(offset, lower, upper)
            apply_joint_target(robot_id, idx, joint_types.get(idx, -1), np.zeros(3, dtype=np.float32), offset)
    for joint_name, euler in ROBOT_STAND_EULER.items():
        idx = joint_index.get(joint_name)
        if idx is not None:
            apply_joint_target(robot_id, idx, joint_types.get(idx, -1), euler, 0.0)
    robot_edges = build_robot_tree(robot_id)
    skeleton_line_ids = []

    if args.print_robot_joints:
        print("Robot joints:")
        for name in sorted(joint_index.keys()):
            print(name)
        return

    selected = 0
    last_text_id = None
    last_debug_log = 0.0
    base_pos_lock = upright_base_pos
    base_orn_lock = upright_base_orn

    def update_hud():
        nonlocal last_text_id
        if last_text_id is not None:
            p.removeUserDebugItem(last_text_id)
        joint_name = CONTROL_JOINTS[selected]
        msg = (
            f"Joint: {joint_name} | select:[ ] | pitch:Up/Down or U/J | "
            f"yaw:Left/Right or H/K | roll:Q/E or Y/I | R reset | 0 reset all"
        )
        last_text_id = p.addUserDebugText(msg, [0, 0, 1.8], textSize=1.2)

    update_hud()

    while True:
        keys = p.getKeyboardEvents()
        if keys:
            if args.debug_keys:
                # Print compact key diagnostics for troubleshooting focus/input issues.
                print("[keys]", sorted(keys.items()))
            if key_triggered(keys, ord("[")) or key_triggered(keys, ord(",")) or key_triggered(keys, ord("n")) or key_triggered(keys, ord("N")):
                selected = (selected - 1) % len(CONTROL_JOINTS)
                update_hud()
                print(f"[select] {selected}: {CONTROL_JOINTS[selected]}")
            if key_triggered(keys, ord("]")) or key_triggered(keys, ord(".")) or key_triggered(keys, ord("m")) or key_triggered(keys, ord("M")):
                selected = (selected + 1) % len(CONTROL_JOINTS)
                update_hud()
                print(f"[select] {selected}: {CONTROL_JOINTS[selected]}")
            if key_triggered(keys, ord("0")):
                body_pose[:] = 0
            if key_triggered(keys, ord("r")) or key_triggered(keys, ord("R")):
                body_pose[joint_to_idx[CONTROL_JOINTS[selected]]] = 0

            delta = np.zeros(3, dtype=np.float32)
            if key_is_down(keys, p.B3G_UP_ARROW) or key_is_down(keys, ord("u")) or key_is_down(keys, ord("U")):
                delta[0] += STEP_RAD
            if key_is_down(keys, p.B3G_DOWN_ARROW) or key_is_down(keys, ord("j")) or key_is_down(keys, ord("J")):
                delta[0] -= STEP_RAD
            if key_is_down(keys, p.B3G_LEFT_ARROW) or key_is_down(keys, ord("h")) or key_is_down(keys, ord("H")):
                delta[1] += STEP_RAD
            if key_is_down(keys, p.B3G_RIGHT_ARROW) or key_is_down(keys, ord("k")) or key_is_down(keys, ord("K")):
                delta[1] -= STEP_RAD
            if key_is_down(keys, ord("q")) or key_is_down(keys, ord("Q")) or key_is_down(keys, ord("y")) or key_is_down(keys, ord("Y")):
                delta[2] += STEP_RAD
            if key_is_down(keys, ord("e")) or key_is_down(keys, ord("E")) or key_is_down(keys, ord("i")) or key_is_down(keys, ord("I")):
                delta[2] -= STEP_RAD
            if np.any(delta):
                idx = joint_to_idx[CONTROL_JOINTS[selected]]
                body_pose[idx] += delta
                body_pose[idx] = np.array(
                    [clamp(v, -2.0, 2.0) for v in body_pose[idx]], dtype=np.float32
                )
        if args.self_test:
            t = time.time()
            for jn in CONTROL_JOINTS:
                idx = joint_to_idx[jn]
                body_pose[idx, 0] = 0.35 * np.sin(t * 1.6 + idx * 0.4)

        # Forward SMPL-X (not used for mesh rendering here, but validates the model path)
        # smplx expects torch tensors
        _ = model(
            global_orient=torch.from_numpy(global_orient).float(),
            body_pose=torch.from_numpy(body_pose[None, ...]).float(),
        )

        # Retarget to robot joints
        for smpl_joint, robot_name, axis, scale in RETARGET_MAP:
            if smpl_joint not in joint_to_idx:
                continue
            smpl_idx = joint_to_idx[smpl_joint]
            aa = body_pose[smpl_idx]
            robot_idx = robot_joint_targets.get(robot_name)
            if robot_idx is None:
                robot_idx = resolve_robot_joint(joint_index, robot_name)
            if robot_idx is None:
                continue
            jt = joint_types.get(robot_idx, -1)
            if jt == p.JOINT_SPHERICAL:
                base_euler = ROBOT_STAND_EULER.get(robot_name, np.zeros(3, dtype=np.float32))
                target_euler = base_euler + (aa * scale)
                apply_joint_target(robot_id, robot_idx, jt, target_euler, 0.0)
            elif jt == p.JOINT_REVOLUTE:
                angle = float(np.dot(axis, aa)) * scale
                angle += ROBOT_STAND_SCALAR.get(robot_name, 0.0)
                lower, upper = joint_limits.get(robot_idx, (-np.inf, np.inf))
                angle = clamp_to_joint_limits(angle, lower, upper)
                apply_joint_target(robot_id, robot_idx, jt, np.zeros(3, dtype=np.float32), angle)

        # Keep base fixed for a pure retargeting POC (no balance/physics).
        p.resetBasePositionAndOrientation(robot_id, base_pos_lock, base_orn_lock)

        # Draw a debug skeleton overlay that stays visible even when mesh rendering glitches.
        for line_idx, (parent_idx, child_idx) in enumerate(robot_edges):
            parent_pos = get_link_world_pos(robot_id, parent_idx)
            child_pos = get_link_world_pos(robot_id, child_idx)
            replace_id = skeleton_line_ids[line_idx] if line_idx < len(skeleton_line_ids) else -1
            new_id = p.addUserDebugLine(
                parent_pos,
                child_pos,
                lineColorRGB=[1.0, 0.2, 0.2],
                lineWidth=3.0,
                lifeTime=0.0,
                replaceItemUniqueId=replace_id,
            )
            if line_idx < len(skeleton_line_ids):
                skeleton_line_ids[line_idx] = new_id
            else:
                skeleton_line_ids.append(new_id)

        now = time.time()
        if now - last_debug_log > 2.0:
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
            print(f"[debug] base_pos={base_pos}")
            if not np.all(np.isfinite(base_pos)):
                print("[debug] detected NaN base pose, resetting robot state")
                p.resetBasePositionAndOrientation(robot_id, [0, 0, 1.0], [0, 0, 0, 1])
                for j in range(p.getNumJoints(robot_id)):
                    p.resetJointState(robot_id, j, targetValue=0.0, targetVelocity=0.0)
            last_debug_log = now

        p.stepSimulation()
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
