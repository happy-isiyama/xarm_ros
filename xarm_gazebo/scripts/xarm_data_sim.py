#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, JointState
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np
import cv2
import os
import threading

class XarmRLDSDatasetBuilder:
    def __init__(self):
        rospy.init_node('xarm_rlds_dataset_builder')
        self.bridge = CvBridge()

        # ディレクトリの設定
        self.image_dir = '/home/yuki/dataset/images'
        os.makedirs(self.image_dir, exist_ok=True)

        # トピックの購読
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback)
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        # データ格納用
        self.observations = {'images': [], 'joint_states': [], 'model_states': []}
        self.actions = []
        self.rewards = []
        self.is_terminal = []
        self.step_counter = 0  # ステップカウンタを追加

        # シャットダウンフラグの初期化
        self.exit_requested = False

        # ステップ進行用のロック
        self.step_lock = threading.Lock()
        self.step_event = threading.Event()

        # 初期化スレッドの起動
        self.init_thread = threading.Thread(target=self.step_control_loop)
        self.init_thread.start()

    def step_control_loop(self):
        while not self.exit_requested:
            input("次のステップに進むには、Enterキーを押してください。")
            if self.exit_requested:
                break
            # ここでステップを進める処理を行います
            self.step_event.set()  # ステップが進んだことを知らせる
            self.step_event.clear()  # イベントをクリアして次の待機状態へ


    def image_callback(self, msg):
        with self.step_lock:
            if self.step_event.is_set():
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                image_path = os.path.join(self.image_dir, f'image_{self.step_counter}.png')
                cv2.imwrite(image_path, cv_image)  # PNG 形式で画像を保存
                self.observations['images'].append(image_path)  # 保存した画像のパスを記録
                self.step_counter += 1  # ステップカウンタを増やす
                self.step_event.set()  # ステップが進んだことを知らせる

    def joint_states_callback(self, msg):
        with self.step_lock:
            if self.step_event.is_set():
                joint_angles = np.array(msg.position)  # ジョイント角度を取得
                self.observations['joint_states'].append(joint_angles)
                self.step_counter += 1  # ステップカウンタを増やす
                self.step_event.set()  # ステップが進んだことを知らせる

    def model_states_callback(self, msg):
        with self.step_lock:
            if self.step_event.is_set():
                for i, name in enumerate(msg.name):
                    if name == 'xarm':
                        # ロボットアームの位置 (x, y, z) と姿勢 (r, p, y) を取得
                        pos = msg.pose[i].position
                        ori = msg.pose[i].orientation
                        # 四元数からオイラー角に変換
                        rpy = self.quaternion_to_euler(ori)
                        end_effector_opening = self.get_end_effector_state()
                        state = {
                            'position': [pos.x, pos.y, pos.z],
                            'rpy': rpy,
                            'end_effector': end_effector_opening
                        }
                        self.observations['model_states'].append(state)
                        self.step_counter += 1  # ステップカウンタを増やす
                        self.step_event.set()  # ステップが進んだことを知らせる

    def quaternion_to_euler(self, orientation):
        # 四元数からオイラー角 (roll, pitch, yaw) に変換する関数
        import tf.transformations as tf_trans
        return tf_trans.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def get_end_effector_state(self):
        # エンドエフェクタの状態を取得するダミー関数 (実際にはセンサー情報を用いる)
        return np.random.uniform(0, 1)  # 0から1のランダムな値を使用

    def build_tf_dataset(self):
        # TF データセットの構築
        def parse_image_path(image_path):
            image = cv2.imread(image_path)  # PNG 画像を読み込み
            return tf.convert_to_tensor(image, dtype=tf.float32)

        images = [parse_image_path(img_path) for img_path in self.observations['images']]
        joint_states = [tf.convert_to_tensor(js, dtype=tf.float32) for js in self.observations['joint_states']]
        model_states = [tf.convert_to_tensor(ms, dtype=tf.float32) for ms in self.observations['model_states']]

        features = {
            'images': tf.convert_to_tensor(images, dtype=tf.float32),
            'joint_states': tf.convert_to_tensor(joint_states, dtype=tf.float32),
            'model_states': tf.convert_to_tensor(model_states, dtype=tf.float32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.float32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'is_terminal': tf.convert_to_tensor(self.is_terminal, dtype=tf.float32),
        }
        dataset = tf.data.Dataset.from_tensor_slices(features)
        return dataset

    def save_tf_dataset(self, path):
        dataset = self.build_tf_dataset()
        tf.data.experimental.save(dataset, path)
        rospy.loginfo(f"TF dataset saved to {path}")

    def shutdown_hook(self):
        # シャットダウン時の処理
        self.exit_requested = True
        self.step_event.set()  # スレッドが待機から復帰するようにイベントを設定
        self.init_thread.join()  # スレッドの終了を待つ
        success = input("実行が終了しました。成功 (s) か失敗 (f) かを入力してください: ").strip().lower()
        self.record_result(success)
        save_path = '/home/yuki/dataset/my_tf_dataset'
        self.save_tf_dataset(save_path)
        rospy.loginfo("Shutting down...")

    def record_result(self, result):
        # 結果をファイルに記録
        with open('/home/yuki/experiment_results.txt', 'w') as f:
            f.write(f"Experiment Result: {result}\n")

if __name__ == '__main__':
    builder = XarmRLDSDatasetBuilder()
    rospy.on_shutdown(builder.shutdown_hook)  # シャットダウン時にフックを設定
    rospy.spin()  # ROSノードを動作させ続ける
