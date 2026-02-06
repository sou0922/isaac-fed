import os
import asyncio
import json
import numpy as np
from PIL import Image
import cv2
import omni.usd
import omni.kit.app
import omni.timeline

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, Gf, Sdf, UsdSkel, Usd, UsdPhysics
import carb

# Setttings
NUM_ROBOTS = 4
RUN_TIME = 500.0        # 実行時間
CAPTURE_INTERVAL = 1.0  # キャプチャ感覚
FORWARD_SPEED = 10.0    # m/s (スケールに合わせて調整)
ANGULAR_SPEED = 0.0     # rad/s（直進のみ）
SCALE_FACTOR = 15.0     # JetBot スケーリング設定


# 各ロボットの初期配置
# JetBotは地面と平行でなければまっすぐ進まないため、yaw（Z軸回転）のみ指定
# +X方向に進む: yaw = 0
# -X方向に進む: yaw = 180度 (π rad)
# camera_resolution: (width, height) - ロボットごとのカメラ解像度
ROBOT_CONFIGS = [
    {"position": (-40.0, 8.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 50.0, "direction": 1, "camera_resolution": (640, 480)},     # 1台目: +X方向へ
    {"position": (-40.0, 3.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 50.0, "direction": 1, "camera_resolution": (640, 480)},     # 2台目: +X方向へ
    {"position": (40.0, -8.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -50.0, "direction": -1, "camera_resolution": (640, 480)},  # 3台目: -X方向へ
    {"position": (40.0, -3.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -50.0, "direction": -1, "camera_resolution": (640, 480)},  # 4台目: -X方向へ
]

# 速度変化の設定
MIN_SPEED = 5.0   # 最小速度 (m/s)
MAX_SPEED = 15.0  # 最大速度 (m/s)
SPEED_CHANGE_INTERVAL_MIN = 3.0  # 速度変更の最小間隔 (秒)
SPEED_CHANGE_INTERVAL_MAX = 8.0  # 速度変更の最大間隔 (秒)
ACCELERATION_SMOOTHNESS = 0.02   # 加減速の滑らかさ (0-1, 小さいほど滑らか)

BASE_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed/robot_images"
VIDEO_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed"

# 人のセマンティックラベル（バウンディングボックス用）
PERSON_SEMANTIC_LABEL = "Human"

# JetBot USDパス
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    raise RuntimeError("Assets root path not found")
JETBOT_USD = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
print(f"JetBot USD path: {JETBOT_USD}")


async def main():
    # 初期化
    print("Initializing World on current stage...")

    # 既存のJetBotプリムと上空カメラを削除（再実行時の衝突を回避）
    stage = omni.usd.get_context().get_stage()
    if stage:
        for i in range(NUM_ROBOTS):
            prim_path = f"/World/JetBot_{i:02d}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim_path)
        # 上空カメラも削除
        topdown_prim = stage.GetPrimAtPath("/World/TopDownCamera")
        if topdown_prim and topdown_prim.IsValid():
            stage.RemovePrim("/World/TopDownCamera")

    World.clear_instance()
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    
    # Z座標はスケールを考慮して設定（車輪が地面に接する高さ）
    # 車輪半径 * スケール = 地面からボディ中心までの高さ
    wheel_radius_scaled = 0.03 * SCALE_FACTOR
    ground_z = wheel_radius_scaled
    print(f"[KINEMATIC] Robot Z position fixed at {ground_z:.2f}m (wheel radius scaled)")
    
    stage = omni.usd.get_context().get_stage()

    # ロボット生成
    print(f"Creating robots with scale factor {SCALE_FACTOR}...")
    robots = []

    for i in range(NUM_ROBOTS):
        robot_id = f"jetbot_{i:02d}"
        robot_path = f"/World/JetBot_{i:02d}"
        config = ROBOT_CONFIGS[i]

        pos = config["position"]
        position = np.array([pos[0], pos[1], ground_z])
        
        yaw_rad = np.deg2rad(config["yaw_deg"])
        orientation = euler_angles_to_quat(np.array([0, 0, yaw_rad]))

        wheeled_robot = world.scene.add(
            WheeledRobot(
                prim_path=robot_path,
                name=robot_id,
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=JETBOT_USD,
                position=position,
                orientation=orientation,
            )
        )

        save_dir = os.path.join(BASE_SAVE_DIR, f"robot_{i:02d}")
        os.makedirs(save_dir, exist_ok=True)

        robots.append({
            "id": robot_id,
            "path": robot_path,
            "wheeled_robot": wheeled_robot,
            "camera": None,
            "save_dir": save_dir,
            "frame": 0,
            "config_idx": i,  # 設定インデックス（再配置用）
            "direction": config["direction"],  # 1 = 正方向(+x)、-1 = 逆方向(-x)
            "turn_at_x": config["turn_at_x"],  # 折り返しX座標
            "yaw_deg": config["yaw_deg"],  # 現在のyaw角度
            "target_z": ground_z,  # 目標Z座標（常に0）
            "initial_y": pos[1],  # 初期Y座標（折り返し時に反転）
            "current_speed": FORWARD_SPEED,  # 現在の速度
            "target_speed": FORWARD_SPEED,   # 目標速度
            "next_speed_change_time": 2.0 + np.random.uniform(SPEED_CHANGE_INTERVAL_MIN, SPEED_CHANGE_INTERVAL_MAX),  # 次の速度変更時刻
        })
        
        print(f"[CREATED] {robot_id} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), yaw={config['yaw_deg']:.0f}°")

    # ロボットにスケール適用
    print("Applying scale to robots...")
    for r in robots:
        prim = stage.GetPrimAtPath(r["path"])
        if prim and prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            # 既存のスケールオペレーションを探す or 追加
            scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
            if scale_ops:
                scale_ops[0].Set(Gf.Vec3f(SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR))
            else:
                xformable.AddScaleOp().Set(Gf.Vec3f(SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR))
            print(f"[SCALE] {r['id']} scaled by {SCALE_FACTOR}x")

    print(f"\n{NUM_ROBOTS} robots scaled.")

    # 上空カメラ（TopView）
    print("Creating top-down camera at (0, 0, 300)...")
    topdown_cam_path = "/World/TopDownCamera"
    # 既存のカメラを削除
    existing_cam = stage.GetPrimAtPath(topdown_cam_path)
    if existing_cam and existing_cam.IsValid():
        stage.RemovePrim(topdown_cam_path)

    # World リセット（async版）
    print("Resetting world...")
    await world.reset_async()
    print("World reset complete.")
    
    # world.reset後にロボットの位置を再設定
    print("Repositioning robots after world reset...")
    for r in robots:
        config = ROBOT_CONFIGS[r["config_idx"]]
        pos = config["position"]
        position = np.array([pos[0], pos[1], ground_z])
        yaw_rad = np.deg2rad(config["yaw_deg"])
        orientation = euler_angles_to_quat(np.array([0, 0, yaw_rad]))
        
        # WheeledRobotの位置を再設定
        r["wheeled_robot"].set_world_pose(position=position, orientation=orientation)
        
        # ホイールジョイントを初期化（0度）
        wr = r["wheeled_robot"]
        if wr.num_dof >= 2:
            wr.set_joint_positions(np.array([0.0, 0.0]))
        
        print(f"[REPOSITION] {r['id']} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), yaw={config['yaw_deg']:.0f}°")

    # Camera初期化（world.reset()の後）
    print("Initializing robot cameras with bounding box detection...")
    for r in robots:
        config = ROBOT_CONFIGS[r["config_idx"]]
        cam_resolution = config.get("camera_resolution", (640, 480))
        cam_path = f"{r['path']}/chassis/fed_camera"
        cam = Camera(
            prim_path=cam_path,
            resolution=cam_resolution,
            frequency=30.0,
            translation=np.array([0.15, 0.0, 0.1]),
        )
        cam.initialize()
        
        # Bounding Box 2D Tightアノテーターを追加
        cam.add_bounding_box_2d_tight_to_frame(init_params={"semanticTypes": ["class"]})
        
        r["camera"] = cam
        print(f"[CAMERA] {r['id']} initialized at {cam_path} with resolution {cam_resolution} and bounding_box_2d_tight")

    # 上空カメラ初期化
    print("Initializing top-down camera...")
    # カメラを作成（向きは後で設定）
    topdown_camera = Camera(
        prim_path=topdown_cam_path,
        resolution=(1920, 1080),
        frequency=30.0,
        translation=np.array([0.0, 0.0, 220.0]),
    )
    topdown_camera.initialize()
    
    # SetLookAtでカメラの向き調整
    cam_prim = stage.GetPrimAtPath(topdown_cam_path)
    if cam_prim and cam_prim.IsValid():
        eye = Gf.Vec3d(0.0, 0.0, 220.0)    # カメラ位置
        target = Gf.Vec3d(0.0, 0.0, 0.0)   # 地面中心
        up_axis = Gf.Vec3d(0.0, 1.0, 0.0)  # Y軸が上（画像の上方向）
        look_at_matrix = Gf.Matrix4d().SetLookAt(eye, target, up_axis)
        look_at_quatd = look_at_matrix.GetInverse().ExtractRotation().GetQuat()
        
        # XformOpを取得または作成してorientを設定
        xformable = UsdGeom.Xformable(cam_prim)
        orient_attr = cam_prim.GetAttribute("xformOp:orient")
        if orient_attr:
            # 属性の型を確認してGfQuatdを使用
            orient_attr.Set(Gf.Quatd(look_at_quatd))
        else:
            # orient属性がない場合は追加
            orient_op = xformable.AddOrientOp()
            orient_op.Set(Gf.Quatd(look_at_quatd))
        print(f"[CAMERA] Top-down camera looking at ground with SetLookAt")
    print(f"[CAMERA] Top-down camera initialized at {topdown_cam_path}")

    # サイドカメラ(SideView)初期化
    print("Initializing side camera...")
    side_cam_path = "/World/SideCamera"
    # 既存のカメラを削除
    existing_side_cam = stage.GetPrimAtPath(side_cam_path)
    if existing_side_cam and existing_side_cam.IsValid():
        stage.RemovePrim(side_cam_path)
    
    # カメラを作成（位置を後ろに移動して引いた画に）
    side_camera = Camera(
        prim_path=side_cam_path,
        resolution=(1920, 1080),
        frequency=30.0,
        translation=np.array([-70.0, 0.0, 9.0]),  # より後ろ、上、高い位置に
    )
    side_camera.initialize()
    
    # カメラの向きを設定（既存のorient操作を削除してrotateXYZを設定）
    side_cam_prim = stage.GetPrimAtPath(side_cam_path)
    if side_cam_prim and side_cam_prim.IsValid():
        xformable = UsdGeom.Xformable(side_cam_prim)
        
        # 既存のすべてのXform操作を取得
        xform_ops = xformable.GetOrderedXformOps()
        
        # orient操作を削除
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                xformable.GetPrim().RemoveProperty(op.GetName())
                print(f"[CAMERA] Removed existing orient operation")
        
        # rotateXYZ操作を探す or 追加
        xform_ops = xformable.GetOrderedXformOps()  # 再取得
        rotate_ops = [op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
        if rotate_ops:
            rotate_ops[0].Set(Gf.Vec3d(83.0, 0.0, -90.0))
        else:
            rotate_op = xformable.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3d(83.0, 0.0, -90.0))
        print(f"[CAMERA] Side camera rotation set to (83, 0, -90)")
    print(f"[CAMERA] Side camera initialized at {side_cam_path}")

    # カメラ設定後、選択状態をクリアして地面のハイライトを解除
    try:
        selection = omni.usd.get_context().get_selection()
        selection.clear_selected_prim_paths()
        print("[CAMERA] Selection cleared to prevent ground highlight")
    except Exception as e:
        print(f"[WARNING] Could not clear selection: {e}")

    # VideoWriter初期化（メモリ節約のため直接書き込み）
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
    topdown_video_path = os.path.join(VIDEO_SAVE_DIR, "simulation_topdown.mp4")
    side_video_path = os.path.join(VIDEO_SAVE_DIR, "simulation_side.mp4")
    
    # 解像度取得
    tw, th = topdown_camera.get_resolution()
    sw, sh = side_camera.get_resolution()
    
    # VideoWriter作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    topdown_writer = cv2.VideoWriter(topdown_video_path, fourcc, 60.0, (tw, th))
    side_writer = cv2.VideoWriter(side_video_path, fourcc, 60.0, (sw, sh))
    
    if not topdown_writer.isOpened():
        print("[ERROR] Failed to open topdown video writer")
    else:
        print(f"[VIDEO] Topdown video writer opened: {topdown_video_path}")
    
    if not side_writer.isOpened():
        print("[ERROR] Failed to open side video writer")
    else:
        print(f"[VIDEO] Side video writer opened: {side_video_path}")
    
    video_frame_count = 0

    scaled_wheel_radius = 0.03 * SCALE_FACTOR
    scaled_wheel_base = 0.1125 * SCALE_FACTOR
    print(f"Robot parameters: wheel_radius={scaled_wheel_radius:.2f}m, wheel_base={scaled_wheel_base:.2f}m")

    print("Checking robot joints and initial orientation...")
    for r in robots:
        wr = r["wheeled_robot"]
        pos, quat = wr.get_world_pose()
        print(f"[DEBUG] {r['id']} dof_names: {wr.dof_names}, num_dof: {wr.num_dof}")
        print(f"        Initial pose: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    print("Starting simulation")

    frame_count = 0
    last_capture_time = 0.0
    last_debug_time = 0.0
    warmup_done = False
    
    # 移動速度 (m/frame) - フレームレートに依存
    # 60fpsを想定、FORWARD_SPEED m/sなので、1フレームあたり FORWARD_SPEED/60 m
    dt = 1.0 / 60.0  # フレーム間隔（約定）

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/motionBlur/enabled", False)
    settings.set("/rtx/taa/enabled", False)

    while world.is_playing():
        await omni.kit.app.get_app().next_update_async()

        current_time = world.current_time
        frame_count += 1

        # ウォームアップ（最初の2秒はロボットを安定させる）
        if current_time < 2.0:
            continue
        
        # ウォームアップ完了メッセージ（一度だけ）
        if not warmup_done:
            print("\n=== WARMUP COMPLETE ===")
            print("Initial robot positions:")
            for r in robots:
                pos, quat = r["wheeled_robot"].get_world_pose()
                print(f"  {r['id']}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            print("Starting kinematic robot movement...\n")
            warmup_done = True
            last_capture_time = current_time

        # デバッグ出力（5秒ごと）
        if current_time - last_debug_time >= 5.0:
            print(f"\n[DEBUG t={current_time:.1f}s]")
            for r in robots:
                pos, quat = r["wheeled_robot"].get_world_pose()
                print(f"  {r['id']}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dir={r['direction']}, yaw={r['yaw_deg']:.0f}°, speed={r['current_speed']:.2f}m/s")
            last_debug_time = current_time

        # ロボットのkinematic移動（位置を直接更新して物理を上書き）
        for r in robots:
            # 速度変更のタイミングチェック
            if current_time >= r["next_speed_change_time"]:
                # 新しい目標速度をランダムに設定
                r["target_speed"] = np.random.uniform(MIN_SPEED, MAX_SPEED)
                # 次の速度変更時刻を設定
                r["next_speed_change_time"] = current_time + np.random.uniform(SPEED_CHANGE_INTERVAL_MIN, SPEED_CHANGE_INTERVAL_MAX)
                print(f"[SPEED] {r['id']} new target speed: {r['target_speed']:.2f} m/s")
            
            # 滑らかに加減速（線形補間）
            r["current_speed"] += (r["target_speed"] - r["current_speed"]) * ACCELERATION_SMOOTHNESS
            
            pos, ori = r["wheeled_robot"].get_world_pose()
            current_x = pos[0]
            current_y = pos[1]
            # Z座標は常に目標値に強制設定（重力無視）
            current_z = r["target_z"]
            
            # 端に到達したら初期位置に戻す
            turn_boundary = abs(r["turn_at_x"])
            reset_to_start = False
            
            if r["direction"] == 1 and current_x >= turn_boundary:
                # +X方向に進んでいて端に到達したら初期位置に戻す
                reset_to_start = True
                print(f"[RESET] {r['id']} reached x={current_x:.1f}, resetting to start position")
            elif r["direction"] == -1 and current_x <= -turn_boundary:
                # -X方向に進んでいて端に到達したら初期位置に戻す
                reset_to_start = True
                print(f"[RESET] {r['id']} reached x={current_x:.1f}, resetting to start position")
            
            if reset_to_start:
                # 初期位置に戻す
                config = ROBOT_CONFIGS[r["config_idx"]]
                current_x = config["position"][0]
                r["initial_y"] = config["position"][1]
                # 向きと進行方向は変えない（初期設定のまま維持）
            
            # 新しい位置を計算（X方向に移動、Yはinitial_yを使用、Zは固定）
            # 現在の速度を使用して移動距離を計算
            move_distance = r["current_speed"] * dt * r["direction"]
            new_x = current_x + move_distance
            new_position = np.array([new_x, r["initial_y"], current_z])
            
            # 向きを計算
            target_yaw_rad = np.deg2rad(r["yaw_deg"])
            target_orientation = euler_angles_to_quat(np.array([0, 0, target_yaw_rad]))
            
            # 位置と向きを強制設定（物理を上書き）
            r["wheeled_robot"].set_world_pose(position=new_position, orientation=target_orientation)
            
            # 物理速度をゼロに設定（落下防止）
            r["wheeled_robot"].set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            r["wheeled_robot"].set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            
            # タイヤを回転させる（視覚効果のため）
            # 移動速度に応じてホイールの角速度を計算
            wheel_radius = 0.03 * SCALE_FACTOR  # スケール後の車輪半径
            wheel_angular_velocity = move_distance / wheel_radius  # rad/frame
            # 両方のホイールを同じ速度で回転
            wr = r["wheeled_robot"]
            if wr.num_dof >= 2:
                current_joint_positions = wr.get_joint_positions()
                if current_joint_positions is not None and len(current_joint_positions) >= 2:
                    new_joint_positions = current_joint_positions.copy()
                    # 左右のホイールを回転（正の方向に進む場合は正の回転）
                    new_joint_positions[0] += wheel_angular_velocity  # left wheel
                    new_joint_positions[1] += wheel_angular_velocity  # right wheel
                    # 角度を[-2π, 2π]の範囲に正規化（PhysXエラー回避）
                    new_joint_positions[0] = np.fmod(new_joint_positions[0], 2 * np.pi)
                    new_joint_positions[1] = np.fmod(new_joint_positions[1], 2 * np.pi)
                    wr.set_joint_positions(new_joint_positions)

        # 上空カメラからフレームをキャプチャして直接VideoWriterに書き込み
        topdown_rgba = topdown_camera.get_rgba()
        if topdown_rgba is not None and topdown_rgba.size > 0 and topdown_writer.isOpened():
            topdown_rgba = topdown_rgba.reshape(th, tw, 4)
            topdown_img = topdown_rgba[:, :, :3].astype(np.uint8)
            topdown_writer.write(cv2.cvtColor(topdown_img, cv2.COLOR_RGB2BGR))

        # サイドカメラからフレームをキャプチャして直接VideoWriterに書き込み
        side_rgba = side_camera.get_rgba()
        if side_rgba is not None and side_rgba.size > 0 and side_writer.isOpened():
            side_rgba = side_rgba.reshape(sh, sw, 4)
            side_img = side_rgba[:, :, :3].astype(np.uint8)
            side_writer.write(cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))
            video_frame_count += 1

        # ロボットカメラからの画像キャプチャ（RGBとBounding Box）
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            for r in robots:
                cam = r["camera"]
                if cam is None:
                    continue

                rgba = cam.get_rgba()
                if rgba is None or rgba.size == 0:
                    continue

                w, h = cam.get_resolution()
                rgba = rgba.reshape(h, w, 4)
                img = rgba[:, :, :3].astype(np.uint8)

                # RGB画像の保存パス
                rgb_filepath = os.path.join(
                    r["save_dir"],
                    f"frame_{r['frame']:04d}.png"
                )
                
                # Bounding Boxデータの保存パス（同じフォルダに保存）
                bbox_filepath = os.path.join(
                    r["save_dir"],
                    f"frame_{r['frame']:04d}_bbox.json"
                )

                # 現在位置と回転を取得
                pos, quat = r["wheeled_robot"].get_world_pose()

                # RGB画像を保存
                Image.fromarray(img).save(rgb_filepath)
                
                # Bounding Box 2D Tightデータを取得して保存
                try:
                    current_frame = cam.get_current_frame()
                    bbox_data = current_frame.get("bounding_box_2d_tight", {})
                    
                    # バウンディングボックスデータを整形
                    bbox_list = []
                    if bbox_data and "data" in bbox_data:
                        for bbox in bbox_data["data"]:
                            bbox_entry = {
                                "semanticId": int(bbox["semanticId"]),
                                "x_min": int(bbox["x_min"]),
                                "y_min": int(bbox["y_min"]),
                                "x_max": int(bbox["x_max"]),
                                "y_max": int(bbox["y_max"]),
                                "occlusionRatio": float(bbox["occlusionRatio"]),
                                "label": PERSON_SEMANTIC_LABEL,  # すべて"Human"ラベル
                            }
                            bbox_list.append(bbox_entry)
                    
                    # idToLabels情報も取得（セマンティックID→ラベルのマッピング）
                    id_to_labels = bbox_data.get("info", {}).get("idToLabels", {})
                    
                    bbox_result = {
                        "frame_id": r["frame"],
                        "robot_id": r["id"],
                        "robot_position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                        "timestamp": float(current_time),
                        "image_resolution": {"width": w, "height": h},
                        "bounding_boxes": bbox_list,
                        "id_to_labels": {str(k): v for k, v in id_to_labels.items()},
                    }
                    
                    with open(bbox_filepath, 'w') as f:
                        json.dump(bbox_result, f, indent=2)
                    
                    print(f"[SAVED] {r['id']} frame_{r['frame']:04d} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), bboxes={len(bbox_list)}")
                except Exception as e:
                    print(f"[WARNING] Failed to get bounding box data for {r['id']}: {e}")
                    print(f"[SAVED] {r['id']} frame_{r['frame']:04d} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) (RGB only)")

                r["frame"] += 1

            last_capture_time = current_time

        # 終了条件
        if current_time > RUN_TIME:
            print("FINISHED")
            break

    # VideoWriterを閉じる
    print(f"Closing video writers...")
    if topdown_writer.isOpened():
        topdown_writer.release()
        print(f"[VIDEO] Topdown camera saved to {topdown_video_path} ({video_frame_count} frames)")
    
    if side_writer.isOpened():
        side_writer.release()
        print(f"[VIDEO] Side camera saved to {side_video_path} ({video_frame_count} frames)")

    print("SCRIPT END")
    world.stop()


# Script Editorから実行
asyncio.ensure_future(main())
