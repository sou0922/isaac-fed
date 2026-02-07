import os
import asyncio
import json
import numpy as np
from PIL import Image
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
from pxr import UsdGeom, Gf, Sdf, UsdSkel
import carb

# アニメーション関連の拡張機能を有効化
ANIM_EXTENSIONS = [
    "omni.anim.people",
    "omni.anim.timeline",
    "omni.anim.graph.bundle",
    "omni.anim.graph.core",
]
print("Enabling animation extensions...")
for ext in ANIM_EXTENSIONS:
    try:
        enable_extension(ext)
        print(f"  [OK] {ext}")
    except Exception as e:
        print(f"  [SKIP] {ext}: {e}")

# =====================================================
# 設定
# =====================================================
NUM_ROBOTS = 4
RUN_TIME = 50.0              # ウォームアップ2秒 + 実行（人が周回するため長め）
CAPTURE_INTERVAL = 2.0
FORWARD_SPEED = 10.0          # m/s (スケールに合わせて調整)
ANGULAR_SPEED = 0.0          # rad/s（直進のみ）

# 人の歩行設定
PERSON_WALK_DISTANCE = 40.0  # 各辺の歩行距離

# JetBot スケーリング設定
# 元のJetBot横幅は約0.125m、2.5mにするためスケール20倍
SCALE_FACTOR = 15.0

# 各ロボットの初期配置
# JetBotは地面と平行でなければまっすぐ進まないため、yaw（Z軸回転）のみ指定
# +X方向に進む: yaw = 0
# -X方向に進む: yaw = 180度 (π rad)
ROBOT_CONFIGS = [
    {"position": (-50.0, -8.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 50.0, "direction": 1},     # 1台目: +X方向へ
    {"position": (-50.0, -3.0, 0.0), "yaw_deg": 0.0, "turn_at_x": 50.0, "direction": 1},     # 2台目: +X方向へ
    {"position": (50.0, 8.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -50.0, "direction": -1},  # 3台目: -X方向へ
    {"position": (50.0, 3.0, 0.0), "yaw_deg": 180.0, "turn_at_x": -50.0, "direction": -1},  # 4台目: -X方向へ
]

# 人の初期配置設定（4人）
# 全員同じルートを周回：(-20,20) → (-20,-20) → (20,-20) → (20,20) → (-20,20)...
# 初期位置と初期方向を設定して、同じ40m四角形を右回りに周回
PERSON_CONFIGS = [
    {
        "name": "Person_00",
        "position": (-20.0, 20.0, 0.0),
        "yaw_deg": -90.0,      # -Y方向を向く (南向き)
        "walk_direction": 0,   # 0: -Y方向から開始
        "walk_speed": 2.2,     # m/s
        "distance_walked": 0.0,
    },
    {
        "name": "Person_01",
        "position": (-20.0, -20.0, 0.0),
        "yaw_deg": 0.0,        # +X方向を向く (東向き)
        "walk_direction": 1,   # 1: +X方向から開始
        "walk_speed": 2.4,     # m/s
        "distance_walked": 0.0,
    },
    {
        "name": "Person_02",
        "position": (20.0, 20.0, 0.0),
        "yaw_deg": 180.0,      # -X方向を向く (西向き)
        "walk_direction": 3,   # 3: -X方向から開始
        "walk_speed": 2.0,     # m/s
        "distance_walked": 0.0,
    },
    {
        "name": "Person_03",
        "position": (20.0, -20.0, 0.0),
        "yaw_deg": 90.0,       # +Y方向を向く (北向き)
        "walk_direction": 2,   # 2: +Y方向から開始
        "walk_speed": 2.6,     # m/s
        "distance_walked": 0.0,
    },
]

BASE_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed/robot_images"
VIDEO_SAVE_DIR = "/home/tamakiokamoto/so/isaac-fed/source/fed"

# 人のセマンティックラベル（統一ラベル）
PERSON_SEMANTIC_LABEL = "person"

# JetBot USDパス
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    raise RuntimeError("Assets root path not found")
JETBOT_USD = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
print(f"JetBot USD path: {JETBOT_USD}")

# 人のUSDパス (Isaac/People/Charactersから4種類)
PERSON_USDS = [
    assets_root_path + "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    assets_root_path + "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    assets_root_path + "/Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
    assets_root_path + "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd",
]
for i, usd_path in enumerate(PERSON_USDS):
    print(f"Person_{i:02d} USD path: {usd_path}")


def get_person_direction_vector(direction_index):
    """歩行方向インデックスから方向ベクトルと向きを返す
    右回り（時計回り）の順序: -Y → +X → +Y → -X → -Y...
    """
    # 0: -Y方向, 1: +X方向, 2: +Y方向, 3: -X方向 (右回り)
    directions = [
        (np.array([0.0, -1.0, 0.0]), -90.0),   # -Y方向、yaw=-90度
        (np.array([1.0, 0.0, 0.0]), 0.0),      # +X方向、yaw=0度
        (np.array([0.0, 1.0, 0.0]), 90.0),     # +Y方向、yaw=90度
        (np.array([-1.0, 0.0, 0.0]), 180.0),   # -X方向、yaw=180度
    ]
    return directions[direction_index % 4]


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
    # 人のキャラクターを追加
    # =================================================
    print("Creating person characters...")
    people = []
    
    for i, person_config in enumerate(PERSON_CONFIGS):
        person_name = person_config["name"]
        person_path = f"/World/{person_name}"
        
        # 既存のプリムを削除
        existing_person = stage.GetPrimAtPath(person_path)
        if existing_person and existing_person.IsValid():
            stage.RemovePrim(person_path)
        
        # 人のキャラクターを追加（各人に異なるUSDを使用）
        person_usd = PERSON_USDS[i % len(PERSON_USDS)]
        person_prim = add_reference_to_stage(
            usd_path=person_usd,
            prim_path=person_path
        )
        
        # 位置と向きを設定
        pos = person_config["position"]
        yaw_deg = person_config["yaw_deg"]
        
        xformable = UsdGeom.Xformable(person_prim)
        
        # トランスレーション設定
        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if translate_ops:
            translate_ops[0].Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        else:
            xformable.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        
        # 回転設定 (Y軸周りの回転 = yaw)
        orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
        yaw_rad = np.deg2rad(yaw_deg)
        # クォータニオン変換 (Z軸周りの回転)
        quat = euler_angles_to_quat(np.array([0, 0, yaw_rad]))
        if orient_ops:
            orient_ops[0].Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
        else:
            xformable.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
        
        # セマンティックラベルを追加（統一ラベル "person"）
        add_update_semantics(person_prim, PERSON_SEMANTIC_LABEL, "class")
        
        # 歩行状態を管理するデータ構造
        people.append({
            "name": person_name,
            "path": person_path,
            "prim": person_prim,
            "position": np.array([pos[0], pos[1], pos[2]]),
            "yaw_deg": yaw_deg,
            "walk_direction": person_config["walk_direction"],
            "walk_speed": person_config["walk_speed"],
            "distance_walked": 0.0,
        })
        
        print(f"[PERSON] {person_name} created at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), yaw={yaw_deg:.0f}°, speed={person_config['walk_speed']:.1f}m/s, label='{PERSON_SEMANTIC_LABEL}'")

    print(f"{len(people)} person(s) created.")

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
    print("Initializing robot cameras with bounding box detection...")
    for r in robots:
        cam_path = f"{r['path']}/chassis/fed_camera"
        cam = Camera(
            prim_path=cam_path,
            resolution=(640, 480),
            frequency=30.0,
            translation=np.array([0.15, 0.0, 0.1]),
        )
        cam.initialize()
        
        # Bounding Box 2D Tightアノテーターを追加
        # semanticTypes=["class"]で"class"タイプのセマンティックラベルを持つオブジェクトを検出
        cam.add_bounding_box_2d_tight_to_frame(init_params={"semanticTypes": ["class"]})
        
        r["camera"] = cam
        print(f"[CAMERA] {r['id']} initialized at {cam_path} with bounding_box_2d_tight")

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

    # カメラ設定後、選択状態をクリアして地面のハイライトを解除
    try:
        selection = omni.usd.get_context().get_selection()
        selection.clear_selected_prim_paths()
        print("[CAMERA] Selection cleared to prevent ground highlight")
    except Exception as e:
        print(f"[WARNING] Could not clear selection: {e}")

    # =================================================
    # タイムラインを開始（アニメーション再生のため）
    # =================================================
    print("Starting timeline for character animations...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[TIMELINE] Timeline started")

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
            print("Initial person positions:")
            for p in people:
                print(f"  {p['name']}: pos=({p['position'][0]:.2f}, {p['position'][1]:.2f}, {p['position'][2]:.2f}), yaw={p['yaw_deg']:.0f}°, speed={p['walk_speed']:.1f}m/s")
            print("Starting kinematic robot and person movement...\n")
            warmup_done = True
            last_capture_time = current_time

        # デバッグ出力（5秒ごと）
        if current_time - last_debug_time >= 5.0:
            print(f"\n[DEBUG t={current_time:.1f}s]")
            for r in robots:
                pos, quat = r["wheeled_robot"].get_world_pose()
                print(f"  {r['id']}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dir={r['direction']}, yaw={r['yaw_deg']:.0f}°")
            for p in people:
                print(f"  {p['name']}: pos=({p['position'][0]:.2f}, {p['position'][1]:.2f}, {p['position'][2]:.2f}), dir_idx={p['walk_direction']}, walked={p['distance_walked']:.1f}m, speed={p['walk_speed']:.1f}m/s")
            last_debug_time = current_time

        # ロボットのkinematic移動（物理を無視して位置を直接更新）
        for r in robots:
            pos, _ = r["wheeled_robot"].get_world_pose()
            current_x = pos[0]
            current_y = pos[1]
            current_z = pos[2]
            
            # 折り返し判定（+250で折り返し、-250で折り返し）
            turn_boundary = abs(r["turn_at_x"])
            if r["direction"] == 1 and current_x >= turn_boundary:
                r["direction"] = -1
                r["yaw_deg"] = 180.0
                print(f"[TURN] {r['id']} reached x={current_x:.1f}, turning to -x direction")
            elif r["direction"] == -1 and current_x <= -turn_boundary:
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

        # =================================================
        # 人の歩行更新（kinematic移動）
        # 40m歩く → 右に90度回転 → 繰り返し（時計回り）
        # =================================================
        for p in people:
            # 現在の方向ベクトルと目標yaw角度を取得
            direction_vec, target_yaw_deg = get_person_direction_vector(p["walk_direction"])
            
            # 今フレームの移動距離（個別の歩行速度を使用）
            move_dist = p["walk_speed"] * dt
            p["distance_walked"] += move_dist
            
            # 位置を更新
            p["position"] = p["position"] + direction_vec * move_dist
            p["yaw_deg"] = target_yaw_deg
            
            # 40m歩いたら右に90度回転（方向インデックスを+1、右回り）
            if p["distance_walked"] >= PERSON_WALK_DISTANCE:
                p["walk_direction"] = (p["walk_direction"] + 1) % 4
                p["distance_walked"] = 0.0
                new_direction_vec, new_yaw_deg = get_person_direction_vector(p["walk_direction"])
                print(f"[PERSON TURN] {p['name']} turned right 90°, new direction index={p['walk_direction']}, yaw={new_yaw_deg:.0f}°")
            
            # プリムの位置と向きを更新
            person_prim = p["prim"]
            xformable = UsdGeom.Xformable(person_prim)
            
            # トランスレーション更新
            translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            if translate_ops:
                translate_ops[0].Set(Gf.Vec3d(float(p["position"][0]), float(p["position"][1]), float(p["position"][2])))
            
            # 向き更新
            yaw_rad = np.deg2rad(p["yaw_deg"])
            quat = euler_angles_to_quat(np.array([0, 0, yaw_rad]))
            orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
            if orient_ops:
                orient_ops[0].Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

        # 上空カメラからフレームをキャプチャ（毎フレーム）
        topdown_rgba = topdown_camera.get_rgba()
        if topdown_rgba is not None and topdown_rgba.size > 0:
            tw, th = topdown_camera.get_resolution()
            topdown_rgba = topdown_rgba.reshape(th, tw, 4)
            topdown_img = topdown_rgba[:, :, :3].astype(np.uint8)
            video_frames.append(topdown_img)

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
                                "label": PERSON_SEMANTIC_LABEL,  # 統一ラベル
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
