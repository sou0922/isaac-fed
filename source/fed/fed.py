import os
import time
import math
import numpy as np
from PIL import Image

import omni.usd
import omni.kit.app
from omni.timeline import get_timeline_interface
import carb.settings

from pxr import UsdPhysics, Sdf

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation

# =====================================================
# 設定
# =====================================================
NUM_ROBOTS = 5
RUN_TIME = 30.0
CAPTURE_INTERVAL = 2.0
FORWARD_SPEED = 5.0
SPACING = 3.0

BASE_SAVE_DIR = os.path.expanduser("~/so/isaacsim_images")

LEFT_WHEEL = "left_wheel_joint"
RIGHT_WHEEL = "right_wheel_joint"

PHYSICS_WARMUP_FRAMES = 120
CAMERA_WARMUP_FRAMES = 30

# =====================================================
# Utils
# =====================================================
def find_articulation_root(stage, robot_root_path):
    root = stage.GetPrimAtPath(robot_root_path)
    if not root or not root.IsValid():
        return None

    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    return None

# =====================================================
# Stage & Physics
# =====================================================
stage = omni.usd.get_context().get_stage()

if not stage.GetPrimAtPath("/World/physicsScene"):
    UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))

# =====================================================
# Render settings
# =====================================================
settings = carb.settings.get_settings()
settings.set("/rtx/post/motionBlur/enabled", False)
settings.set("/rtx/taa/enabled", False)

# =====================================================
# Timeline & App
# =====================================================
timeline = get_timeline_interface()
app = omni.kit.app.get_app()

# =====================================================
# JetBot & Camera
# =====================================================
robots = []

for i in range(NUM_ROBOTS):
    robot_id = f"robot_{i:02d}"
    robot_path = f"/World/JetBot_{i:02d}"

    if not stage.GetPrimAtPath(robot_path):
        add_reference_to_stage(
            usd_path="/Isaac/Sim/Robots/NVIDIA/Jetbot/jetbot.usd",
            prim_path=robot_path
        )
        
        # ステージ更新を待つ
        app.update()

    # 初期配置（位置と向きを変える）
    yaw = i * 0.3
    xform = get_prim_at_path(robot_path)
    
    # プリムが新規作成された場合のみRigidBodyAPIを適用
    if not xform.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(xform)

    # Xformableを使って既存のopsを取得または作成
    from pxr import UsdGeom, Gf
    xformable = UsdGeom.Xformable(xform)
    
    # 既存のtransform opsをクリア
    xformable.ClearXformOpOrder()
    
    # 新しいopsを追加
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()
    
    translate_op.Set(Gf.Vec3d(i * SPACING, i * 0.5, 0.0))
    orient_op.Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))

    # Camera（JetBotに親子付け）
    cam = Camera(
        prim_path=f"{robot_path}/camera",
        position=(0.15, 0.0, 0.4),
        resolution=(640, 480),
        frequency=30.0
    )
    cam.initialize()

    save_dir = os.path.join(BASE_SAVE_DIR, robot_id)
    os.makedirs(save_dir, exist_ok=True)

    robots.append({
        "id": robot_id,
        "path": robot_path,
        "art": None,
        "controller": None,
        "camera": cam,
        "save_dir": save_dir,
        "frame": 0
    })

print("READY → PRESS ▶ PLAY")

initialized = False
physics_frames = 0
camera_frames = 0
camera_ready = False
start_time = None
last_capture = 0.0

# =====================================================
# Main Loop
# =====================================================
while app.is_running():
    app.update()

    if not timeline.is_playing():
        time.sleep(0.1)
        continue

    physics_frames += 1
    if not initialized and physics_frames < PHYSICS_WARMUP_FRAMES:
        continue

    # ---- Articulation 初期化 ----
    if not initialized:
        print("INITIALIZING ARTICULATIONS")

        for r in robots:
            art_root = find_articulation_root(stage, r["path"])
            if art_root is None:
                print(f"[ERROR] ArticulationRoot not found: {r['id']}")
                continue

            art = Articulation(art_root)
            art.initialize()

            r["art"] = art
            r["controller"] = art.get_articulation_controller()

            print(f"[OK] {r['id']} articulation root:", art_root)

        initialized = True
        start_time = time.time()
        last_capture = start_time
        print("INITIALIZED OK")

    # ---- Camera warmup ----
    if initialized and not camera_ready:
        camera_frames += 1
        if camera_frames < CAMERA_WARMUP_FRAMES:
            continue
        camera_ready = True
        print("CAMERA READY")

    # ---- 終了 ----
    if time.time() - start_time > RUN_TIME:
        print("FINISHED")
        break

    # ---- 前進制御 ----
    for r in robots:
        if r["controller"] is None:
            continue
        r["controller"].set_joint_velocities({
            LEFT_WHEEL: FORWARD_SPEED,
            RIGHT_WHEEL: FORWARD_SPEED
        })

    # ---- 撮影 ----
    if time.time() - last_capture >= CAPTURE_INTERVAL:
        for r in robots:
            rgba = r["camera"].get_rgba()
            if rgba is None or rgba.size == 0:
                continue

            w, h = r["camera"].get_resolution()
            rgba = rgba.reshape(h, w, 4)
            img = rgba[:, :, :3].astype(np.uint8)

            path = os.path.join(
                r["save_dir"],
                f"frame_{r['frame']:04d}.png"
            )

            Image.fromarray(img).save(path)
            print(f"[SAVED] {r['id']} → {path}")
            r["frame"] += 1

        last_capture = time.time()

print("SCRIPT END")
