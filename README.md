conda activate sim2rl-py311


# SMPL-X to PyBullet Humanoid POC

This is a minimal real-time motion retargeting loop:

```
Keyboard -> SMPL-X pose -> retarget map -> PyBullet humanoid
```

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Download SMPL-X models from the official site and place them in:

```
assets/smplx/
```

If your download unzips to `models/`, keep that structure. The script accepts either `assets/smplx/` or `assets/smplx/models/`.
If you only have a single file like `SMPLX_NEUTRAL_2020.npz`, place it in `assets/smplx/` and the script will auto-detect it.

## Run

```
python poc/run.py
```

Optional: list robot joints to adjust the retarget map.

```
python poc/run.py --print-robot-joints
```

If your model file is somewhere else:

```
python poc/run.py --model-file /absolute/path/to/SMPLX_NEUTRAL_2020.npz
```

## Controls

- `[` / `]` : select SMPL-X joint
- `Arrow keys` : pitch/yaw
- `Q` / `E` : roll
- `R` : reset selected joint
- `0` : reset all joints

## Notes

- The retarget map is defined in `poc/run.py` (search `RETARGET_MAP`).
- This uses a generic PyBullet humanoid URDF. You can swap `--urdf` to another humanoid model.
