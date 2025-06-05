# iASL-E2ETENTH-2025

## setup dependencies
```bash
## ros2
# apt から install
sudo apt install ros-${ROS_DISTRO}-laser-filters # for 2d lidar filter
sudo apt install ros-${ROS_DISTRO}-laser-proc # for urg_node2
sudo apt install ros-${ROS_DISTRO}-ackermann-msgs # for jetracer command
sudo apt install ros-${ROS_DISTRO}-rosbag2-storage-mcap # for rosbag

## python
# for jetracer 
git clone https://github.com/NVIDIA-AI-IOT/jetracer.git
cd jetracer
sudo python3 setup.py install

# for 2d Lidar Graph CUDA
cd python/lidar_graph
pip3 install --user -e .
```

## clone and build
```bash
# clone
cd ~
git clone https://github.com/iASL-Gifu/E2ETENTH-2025-JP.git

## .repos内のパッケージをclone
cd ~/E2ETENTH-2025-JP
vcs import ros2_ws/src < packages.repos

## urg nodeのサブモジュールをclone
cd ros2_ws/src/sensors/urg_node2/
git submodule update --init --recursive

# build
cd ~/E2ETENTH-2025-JP
colcon build --symlink-install
```

## run

### 1. base system
カメラ, 2d Lidar, joy teleop, 2d lidar filter, jetracerを起動する. 
これを起動した後は, カメラ画像, 2d lidarのスキャンがrvizから確認できるはず. 
コントローラの制御, E2Eの制御をjoyで切り替えて操作する. [ボタン配置などの設定はconfigにて(ros2のバージョン humbleとfoxyで少し異なるので注意)](./src/core/joy_manager/config/teleop.param.yaml)

```bash
source install/setup.bash
ros2 launch system_launch base_system.launch.xml
```

### 2. rosbag manager launch 
joy経由でrosbagを取りたい場合にsystem launchに加えてこれもlaunchする. 
defualtでは, R2ボタンで記録開始, L2ボタンで記録停止
[記録するtopicはあらかじめyamlに書いておく](./src/core/bag_manager_py/config/bag_manager.param.yaml)
```
ros2 launch bag_manager_py bag_manager_node.launch.xml 
```

### 3. E2E制御
作成中...