import os
import asyncio
import numpy as np
from PIL import Image
import omni.usd
import omni.kit.app

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
import carb

# =====================================================
# 設定
# =====================================================
NUM_ROBOTS = 5
RUN_TIME = 20.0
CAPTURE_INTERVAL = 2.0
FORWARD_SPEED = 0.3          # m/s
ANGULAR_SPEED = 0.0          # rad/s（直進のみ）
SPACING = 3.0

BASE_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed/robot_images"

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

    # 既存のJetBotプリムを削除（再実行時の衝突を回避）
    stage = omni.usd.get_context().get_stage()
    if stage:
        for i in range(NUM_ROBOTS):
            prim_path = f"/World/JetBot_{i:02d}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim_path)

    World.clear_instance()
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # =================================================
    # ロボット作成（WheeledRobot + create_robot=True）
    # =================================================
    print("Creating robots...")
    robots = []

    for i in range(NUM_ROBOTS):
        robot_id = f"jetbot_{i:02d}"
        robot_path = f"/World/JetBot_{i:02d}"

        position = np.array([i * SPACING, i * 0.5, 0.05])
        yaw_deg = i * (360.0 / NUM_ROBOTS)
        yaw_rad = np.deg2rad(yaw_deg)
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
        })

        print(f"[CREATED] {robot_id} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), yaw: {yaw_deg:.1f}°")

    print(f"\n{NUM_ROBOTS} robots created.")

    # =================================================
    # World リセット（async版）
    # =================================================
    print("Resetting world...")
    await world.reset_async()
    print("World reset complete.")

    # =================================================
    # Camera初期化（world.reset()の後）
    # =================================================
    print("Initializing cameras...")
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
    # コントローラー作成
    # =================================================
    controller = DifferentialController(
        name="diff_controller",
        wheel_radius=0.03,
        wheel_base=0.1125,
    )
    print("Differential controller created.")

    # =================================================
    # ジョイント情報確認
    # =================================================
    print("Checking robot joints...")
    for r in robots:
        wr = r["wheeled_robot"]
        print(f"[DEBUG] {r['id']} dof_names: {wr.dof_names}, num_dof: {wr.num_dof}")

    # =================================================
    # シミュレーションループ（async版）
    # =================================================
    print("Starting simulation...")

    frame_count = 0
    last_capture_time = 0.0

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/motionBlur/enabled", False)
    settings.set("/rtx/taa/enabled", False)

    while world.is_playing():
        await omni.kit.app.get_app().next_update_async()

        current_time = world.current_time
        frame_count += 1

        # ウォームアップ（最初の1秒）
        if current_time < 1.0:
            continue

        # ロボット制御（全ロボットを前進させる）
        wheel_action = controller.forward(command=[FORWARD_SPEED, ANGULAR_SPEED])
        for r in robots:
            r["wheeled_robot"].apply_wheel_actions(wheel_action)

        # 画像キャプチャ
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

                # 現在位置を取得
                pos, _ = r["wheeled_robot"].get_world_pose()

                print(f"[SAVED] {r['id']} frame_{r['frame']:04d} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

                Image.fromarray(img).save(filepath)
                r["frame"] += 1

            last_capture_time = current_time

        # 終了条件
        if current_time > RUN_TIME:
            print("FINISHED")
            break

    print("SCRIPT END")
    world.stop()


# Script Editorから実行
asyncio.ensure_future(main())
