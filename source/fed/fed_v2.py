import os
import asyncio
import numpy as np
from PIL import Image
import omni.usd
import omni.kit.app

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, Gf
import carb

# =====================================================
# 設定
# =====================================================
NUM_ROBOTS = 4
RUN_TIME = 62.0              # ウォームアップ2秒 + 実行60秒
CAPTURE_INTERVAL = 2.0
FORWARD_SPEED = 6.0          # m/s (スケールに合わせて調整)
ANGULAR_SPEED = 0.0          # rad/s（直進のみ）

# JetBot スケーリング設定
# 元のJetBot横幅は約0.125m、2.5mにするためスケール20倍
SCALE_FACTOR = 10.0

# 各ロボットの初期配置
# JetBotは地面と平行でなければまっすぐ進まないため、yaw（Z軸回転）のみ指定
# +X方向に進む: yaw = 0
# -X方向に進む: yaw = 180度 (π rad)
ROBOT_CONFIGS = [
    {"position": (-250.0, -8.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 250.0, "direction": 1},     # 1台目: +X方向へ
    {"position": (-250.0, -3.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 250.0, "direction": 1},     # 2台目: +X方向へ
    {"position": (250.0, 8.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -250.0, "direction": -1},  # 3台目: -X方向へ
    {"position": (250.0, 3.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -250.0, "direction": -1},  # 4台目: -X方向へ
]

BASE_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed/robot_images"
VIDEO_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed"

# JetBot USDパス
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    raise RuntimeError("Assets root path not found")
JETBOT_USD = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
print(f"JetBot USD path: {JETBOT_USD}")


async def main():
    # =================================================
    # 現在のステージ上にWorldを初期化
    # =================================================
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
    world.scene.add_default_ground_plane()

    # =================================================
    # ロボット作成（WheeledRobot + create_robot=True）
    # =================================================
    print(f"Creating robots with scale factor {SCALE_FACTOR}...")
    robots = []

    for i in range(NUM_ROBOTS):
        robot_id = f"jetbot_{i:02d}"
        robot_path = f"/World/JetBot_{i:02d}"
        config = ROBOT_CONFIGS[i]

        # 位置設定
        # JetBotの車輪半径は約0.03m、スケール後は 0.03 * SCALEになる
        # 車輪が地面に接するように設定（z_offset = scaled wheel radius）
        pos = config["position"]
        z_offset = 0.03 * SCALE_FACTOR  # 車輪半径分だけ浮かせる（地面接地）
        position = np.array([pos[0], pos[1], pos[2] + z_offset])
        
        # 回転設定: yaw（Z軸回転）のみ
        # JetBotが地面と平行になるように roll=0, pitch=0 に固定
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
            "direction": config["direction"],  # 1 = 正方向(+x)、-1 = 逆方向(-x)
            "turn_at_x": config["turn_at_x"],  # 折り返しX座標
            "yaw_deg": config["yaw_deg"],  # 現在のyaw角度
        })
        
        print(f"[CREATED] {robot_id} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), yaw={config['yaw_deg']:.0f}°")

    # =================================================
    # ロボットにスケール適用
    # =================================================
    print("Applying scale to robots...")
    stage = omni.usd.get_context().get_stage()
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

    print(f"\n{NUM_ROBOTS} robots created and scaled.")

    # =================================================
    # 上空カメラ（ワールド中央上部から見下ろす）
    # =================================================
    print("Creating top-down camera at (0, 0, 300)...")
    topdown_cam_path = "/World/TopDownCamera"
    # 既存のカメラを削除
    existing_cam = stage.GetPrimAtPath(topdown_cam_path)
    if existing_cam and existing_cam.IsValid():
        stage.RemovePrim(topdown_cam_path)

    # =================================================
    # World リセット（async版）
    # =================================================
    print("Resetting world...")
    await world.reset_async()
    print("World reset complete.")

    # =================================================
    # Camera初期化（world.reset()の後）
    # =================================================
    print("Initializing robot cameras...")
    for r in robots:
        cam_path = f"{r['path']}/chassis/fed_camera"
        cam = Camera(
            prim_path=cam_path,
            resolution=(640, 480),
            frequency=30.0,
            translation=np.array([0.15, 0.0, 0.1]),
        )
        cam.initialize()
        r["camera"] = cam
        print(f"[CAMERA] {r['id']} initialized at {cam_path}")

    # =================================================
    # 上空カメラ初期化
    # =================================================
    print("Initializing top-down camera...")
    # カメラを作成（向きは後で設定）
    topdown_camera = Camera(
        prim_path=topdown_cam_path,
        resolution=(1920, 1080),
        frequency=30.0,
        translation=np.array([0.0, 0.0, 300.0]),
    )
    topdown_camera.initialize()
    
    # SetLookAtを使って地面を向くようにカメラの向きを設定
    cam_prim = stage.GetPrimAtPath(topdown_cam_path)
    if cam_prim and cam_prim.IsValid():
        eye = Gf.Vec3d(0.0, 0.0, 300.0)    # カメラ位置
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

    # 動画用フレームリスト
    video_frames = []

    # =================================================
    # コントローラー作成（kinematic移動では使用しないがログ用に残す）
    # =================================================
    scaled_wheel_radius = 0.03 * SCALE_FACTOR
    scaled_wheel_base = 0.1125 * SCALE_FACTOR
    print(f"Robot parameters: wheel_radius={scaled_wheel_radius:.2f}m, wheel_base={scaled_wheel_base:.2f}m")

    # =================================================
    # ジョイント情報確認
    # =================================================
    print("Checking robot joints and initial orientation...")
    for r in robots:
        wr = r["wheeled_robot"]
        pos, quat = wr.get_world_pose()
        print(f"[DEBUG] {r['id']} dof_names: {wr.dof_names}, num_dof: {wr.num_dof}")
        print(f"        Initial pose: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")

    # =================================================
    # シミュレーションループ（kinematic移動版）
    # =================================================
    print("Starting simulation (kinematic movement)...")

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

        # デバッグ出力（1秒ごと）
        if current_time - last_debug_time >= 1.0:
            print(f"\n[DEBUG t={current_time:.1f}s]")
            for r in robots:
                pos, quat = r["wheeled_robot"].get_world_pose()
                print(f"  {r['id']}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dir={r['direction']}, yaw={r['yaw_deg']:.0f}°")
            last_debug_time = current_time

        # ロボットのkinematic移動（物理を無視して位置を直接更新）
        for r in robots:
            pos, _ = r["wheeled_robot"].get_world_pose()
            current_x = pos[0]
            current_y = pos[1]
            current_z = pos[2]
            
            # 折り返し判定
            if r["direction"] == 1 and current_x >= r["turn_at_x"]:
                r["direction"] = -1
                r["yaw_deg"] = 180.0
                print(f"[TURN] {r['id']} reached x={current_x:.1f}, turning to -x direction")
            elif r["direction"] == -1 and current_x <= r["turn_at_x"]:
                r["direction"] = 1
                r["yaw_deg"] = 0.0
                print(f"[TURN] {r['id']} reached x={current_x:.1f}, turning to +x direction")
            
            # 新しい位置を計算（X方向のみ移動、YとZは固定）
            move_distance = FORWARD_SPEED * dt * r["direction"]
            new_x = current_x + move_distance
            new_position = np.array([new_x, current_y, current_z])
            
            # 向きを計算
            target_yaw_rad = np.deg2rad(r["yaw_deg"])
            target_orientation = euler_angles_to_quat(np.array([0, 0, target_yaw_rad]))
            
            # 位置と向きを直接設定（kinematic移動）
            r["wheeled_robot"].set_world_pose(position=new_position, orientation=target_orientation)

        # 上空カメラからフレームをキャプチャ（毎フレーム）
        topdown_rgba = topdown_camera.get_rgba()
        if topdown_rgba is not None and topdown_rgba.size > 0:
            tw, th = topdown_camera.get_resolution()
            topdown_rgba = topdown_rgba.reshape(th, tw, 4)
            topdown_img = topdown_rgba[:, :, :3].astype(np.uint8)
            video_frames.append(topdown_img)

        # ロボットカメラからの画像キャプチャ
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

                filepath = os.path.join(
                    r["save_dir"],
                    f"frame_{r['frame']:04d}.png"
                )

                # 現在位置と回転を取得
                pos, quat = r["wheeled_robot"].get_world_pose()

                print(f"[SAVED] {r['id']} frame_{r['frame']:04d} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dir={r['direction']}, yaw={r['yaw_deg']:.0f}°")

                Image.fromarray(img).save(filepath)
                r["frame"] += 1

            last_capture_time = current_time

        # 終了条件
        if current_time > RUN_TIME:
            print("FINISHED")
            break

    # =================================================
    # 動画を保存
    # =================================================
    print(f"Saving video ({len(video_frames)} frames)...")
    if video_frames:
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
        video_saved = False
        
        # 方法1: OpenCV (cv2) でMP4形式を試す (H.264コーデック)
        try:
            import cv2
            video_path = os.path.join(VIDEO_SAVE_DIR, "simulation_topdown.mp4")
            h, w = video_frames[0].shape[:2]
            # avc1またはH264コーデックを試す
            for codec in ['avc1', 'H264', 'mp4v', 'XVID']:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                ext = '.mp4' if codec in ['avc1', 'H264', 'mp4v'] else '.avi'
                video_path = os.path.join(VIDEO_SAVE_DIR, f"simulation_topdown{ext}")
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                if out.isOpened():
                    for frame in video_frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    print(f"[VIDEO] Saved to {video_path} (using OpenCV {codec})")
                    video_saved = True
                    break
                else:
                    print(f"Codec {codec} not available, trying next...")
        except ImportError:
            print("OpenCV not available...")
        except Exception as e:
            print(f"OpenCV failed: {e}")
        
        # 方法2: GIFとして保存（PILで可能）
        if not video_saved:
            try:
                video_path = os.path.join(VIDEO_SAVE_DIR, "simulation_topdown.gif")
                pil_frames = [Image.fromarray(f) for f in video_frames[::3]]  # 3フレームごと（GIFサイズ削減）
                pil_frames[0].save(
                    video_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,  # 100ms per frame
                    loop=0
                )
                print(f"[VIDEO] Saved to {video_path} (as GIF, every 3rd frame)")
                video_saved = True
            except Exception as e:
                print(f"GIF save failed: {e}")
        
        # 方法3: PNGシーケンスとして保存
        if not video_saved:
            print("Saving frames as PNG sequence...")
            video_frames_dir = os.path.join(VIDEO_SAVE_DIR, "topdown_frames")
            os.makedirs(video_frames_dir, exist_ok=True)
            for i, frame in enumerate(video_frames):
                Image.fromarray(frame).save(os.path.join(video_frames_dir, f"frame_{i:04d}.png"))
            print(f"[FRAMES] Saved {len(video_frames)} frames to {video_frames_dir}")
            print("To convert to video: ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

    print("SCRIPT END")
    world.stop()


# Script Editorから実行
asyncio.ensure_future(main())
