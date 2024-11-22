import cv2
import cv2.aruco as aruco
import os

# ArUcoマーカーの種類を指定
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

# 保存先ディレクトリの設定
save_dir = "/home/yuki/catkin_ws/src/xarm_ros/xarm_gazebo/markers"
os.makedirs(save_dir, exist_ok=True)

# 生成するマーカーIDを指定（例: 31～34）
marker_ids = [31, 32, 33, 34]

# 各マーカーを生成して保存
for marker_id in marker_ids:
    # マーカーを生成
    img = aruco.generateImageMarker(aruco_dict, marker_id, 200)  # サイズ200x200ピクセル
    filename = f"4x4_1000-{marker_id}.png"
    filepath = os.path.join(save_dir, filename)
    
    # 画像を保存
    cv2.imwrite(filepath, img)
    print(f"マーカーID {marker_id} を {filepath} に保存しました。")
