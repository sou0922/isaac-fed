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
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation

# =====================================================
# 設定
# =====================================================
NUM_ROBOTS = 5
RUN_TIME = 20.0
CAPTURE_INTERVAL = 2.0
FORWARD_SPEED = 5.0
SPACING = 3.0

BASE_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed/robot_images"

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

    # JetBotの場合、chassis がArticulationRoot
    chassis_path = f"{robot_root_path}/chassis"
    chassis = stage.GetPrimAtPath(chassis_path)
    if chassis and chassis.IsValid() and chassis.HasAPI(UsdPhysics.ArticulationRootAPI):
        return chassis_path
    
    # フォールバック：全体を探索
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

# 全ロボットが存在するかチェック
all_robots_exist = all(
    stage.GetPrimAtPath(f"/World/JetBot_{i:02d}").IsValid() 
    for i in range(NUM_ROBOTS)
)

# 新規作成が必要な場合は作成のみ実行
if not all_robots_exist:
    print("Creating robots...")
    for i in range(NUM_ROBOTS):
        robot_id = f"robot_{i:02d}"
        robot_path = f"/World/JetBot_{i:02d}"
        
        if not stage.GetPrimAtPath(robot_path).IsValid():
            add_reference_to_stage(
                usd_path="/Isaac/Sim/Robots/NVIDIA/Jetbot/jetbot.usd",
                prim_path=robot_path
            )
            print(f"[CREATED] {robot_id}")
    
    print("Waiting for USD references to fully load...")
    # USD参照が完全にロードされるまで待機（シャーシなどのサブプリムも含む）
    for _ in range(240):  # 約4秒
        app.update()
    
    # Payloadを明示的にロード（USD参照の中身を展開）
    print("Loading USD payloads...")
    for i in range(NUM_ROBOTS):
        robot_path = f"/World/JetBot_{i:02d}"
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim.IsValid():
            robot_prim.Load()  # Payloadをロード
            print(f"  Loaded payload for {robot_path}")
    
    # Payloadロード後も少し待機
    for _ in range(120):
        app.update()
    
    print("Robots ready, configuring positions...")
    all_robots_exist = True  # 続行フラグを立てる

# 既存ロボットの設定
if all_robots_exist:
    print(f"Configuring {NUM_ROBOTS} robots...")
    
    # 既存ロボットの設定
    for i in range(NUM_ROBOTS):
        robot_id = f"robot_{i:02d}"
        robot_path = f"/World/JetBot_{i:02d}"

        # 位置と向きを設定 - get_prim_at_pathで既存プリムを取得
        yaw_deg = i * (360.0 / NUM_ROBOTS)
        
        # 既存のプリムを取得して位置を設定
        robot_xform = get_prim_at_path(robot_path)
        if robot_xform:
            from pxr import UsdGeom, Gf
            xformable = UsdGeom.Xformable(robot_xform)
            
            # xformOpsが既に存在する場合はそれを使う、なければ作成
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            orient_op = None
            
            # 既存のopsを探す
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    orient_op = op
            
            # なければ新規作成
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = xformable.AddOrientOp()
            
            # 値を設定
            translate_op.Set(Gf.Vec3d(i * SPACING, i * 0.5, 0.0))
            quat = euler_angles_to_quat(np.array([0, 0, math.radians(yaw_deg)]))
            orient_op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))  # Quatf (float)
            
            print(f"[SETUP] {robot_id} position: ({i * SPACING:.2f}, {i * 0.5:.2f}, 0.00), yaw: {yaw_deg:.1f}°")
        else:
            print(f"[ERROR] {robot_id} prim not found")

        # Camera（JetBotに親子付け）
        cam = Camera(
            prim_path=f"{robot_path}/camera",
            resolution=(640, 480),
            frequency=30.0
        )
        cam.initialize()
        
        # カメラの相対位置を明示的に設定（ロボット座標系）
        from pxr import UsdGeom, Gf
        cam_prim = stage.GetPrimAtPath(f"{robot_path}/camera")
        if cam_prim.IsValid():
            xformable = UsdGeom.Xformable(cam_prim)
            # 既存のopsを探す
            ops = xformable.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            # なければ新規作成
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            # 相対位置を設定
            translate_op.Set(Gf.Vec3d(0.15, 0.0, 0.4))  # ロボット前方、高さ0.4
            print(f"[CAMERA] {robot_id} camera set at relative position (0.15, 0.0, 0.4)")

        save_dir = os.path.join(BASE_SAVE_DIR, robot_id)
        os.makedirs(save_dir, exist_ok=True)

        robots.append({
            "id": robot_id,
            "path": robot_path,
            "chassis_path": f"{robot_path}/chassis",
            "art": None,
            "controller": None,
            "camera": cam,
            "save_dir": save_dir,
            "frame": 0
        })

    print(f"\n{len(robots)} robots configured.")
    print("Starting simulation automatically...")
    
    # 自動的にタイムラインを再生
    timeline.play()
    
    # タイムラインが開始するまで少し待機
    for _ in range(5):
        app.update()
        time.sleep(0.1)
    
    print("SIMULATION STARTED")

    initialized = False
    chassis_ready = False
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
        
        # ---- Chassis有効化待機（シミュレーション開始後） ----
        if not chassis_ready:
            # 60フレームごとにチェック
            if physics_frames % 60 == 0:
                all_chassis_valid = True
                for i in range(NUM_ROBOTS):
                    robot_path = f"/World/JetBot_{i:02d}"
                    chassis_path = f"{robot_path}/chassis"
                    chassis_prim = stage.GetPrimAtPath(chassis_path)
                    
                    if not chassis_prim.IsValid():
                        all_chassis_valid = False
                        # デバッグ：ロボットプリムの子を調査
                        robot_prim = stage.GetPrimAtPath(robot_path)
                        if robot_prim.IsValid():
                            children = list(robot_prim.GetChildren())
                            child_names = [child.GetName() for child in children[:10]]  # 最初の10個
                            print(f"[DEBUG] Frame {physics_frames}: {robot_path} has {len(children)} children: {child_names}")
                        else:
                            print(f"[ERROR] Frame {physics_frames}: {robot_path} is not valid!")
                        print(f"[WAIT] Frame {physics_frames}: Waiting for chassis to become valid...")
                        break
                
                if all_chassis_valid:
                    print(f"[READY] All chassis valid at frame {physics_frames}")
                    chassis_ready = True
                    physics_frames = 0  # リセットしてウォームアップを開始
            continue
        
        if not initialized and physics_frames < PHYSICS_WARMUP_FRAMES:
            continue

        # ---- Articulation 初期化 ----
        if not initialized:
            print("INITIALIZING ARTICULATIONS")

            for r in robots:
                try:
                    # chassis パスで直接初期化
                    chassis_prim = stage.GetPrimAtPath(r["chassis_path"])
                    if not chassis_prim.IsValid():
                        print(f"[ERROR] {r['id']} chassis prim invalid: {r['chassis_path']}")
                        continue
                    
                    art = Articulation(r["chassis_path"])
                    art.initialize()
                    r["art"] = art
                    
                    # ジョイント名を確認
                    joint_names = art.dof_names
                    print(f"[DEBUG] {r['id']} joints: {joint_names}")
                    
                    # ジョイントインデックスを取得
                    try:
                        left_idx = joint_names.index(LEFT_WHEEL)
                        right_idx = joint_names.index(RIGHT_WHEEL)
                        r["left_idx"] = left_idx
                        r["right_idx"] = right_idx
                        print(f"[DEBUG] {r['id']} wheel indices: left={left_idx}, right={right_idx}")
                    except ValueError as e:
                        print(f"[ERROR] {r['id']} joint not found: {e}")
                        continue
                    
                    # 位置確認
                    xform = stage.GetPrimAtPath(r["path"])
                    translate_attr = xform.GetAttribute("xformOp:translate")
                    if translate_attr:
                        pos = translate_attr.Get()
                        print(f"[OK] {r['id']} initialized at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    else:
                        print(f"[OK] {r['id']} initialized")
                except Exception as e:
                    print(f"[ERROR] {r['id']} articulation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

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
            if r["art"] is None or "left_idx" not in r or "right_idx" not in r:
                continue
            
            # 速度配列を作成（全ジョイント分）
            num_dof = r["art"].num_dof
            velocities = [0.0] * num_dof
            velocities[r["left_idx"]] = FORWARD_SPEED
            velocities[r["right_idx"]] = FORWARD_SPEED
            
            # 速度を設定
            r["art"].set_joint_velocities(velocities)

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

                # 現在位置を取得して表示
                xform = stage.GetPrimAtPath(r["path"])
                translate_attr = xform.GetAttribute("xformOp:translate")
                if translate_attr:
                    pos = translate_attr.Get()
                    print(f"[SAVED] {r['id']} frame_{r['frame']:04d} at position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                else:
                    print(f"[SAVED] {r['id']} frame_{r['frame']:04d}")
                
                Image.fromarray(img).save(path)
                r["frame"] += 1

            last_capture = time.time()

    print("SCRIPT END")
