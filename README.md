# iASL-E2ETENTH-2025

## setup dependencies
```bash
# apt から install
sudo apt install ros-humble-laser-filters # for 2d lidar filter
sudo apt install ros-humble-rosbag2-storage-mcap # for rosbag
```

## clone and build
```bash
# clone
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/iASL-Gifu/E2ETENTH-2025-JP.git

cd E2ETENTH-2025-JP
vcs import src < packages.repos

cd src/sensors/urg_node2/
git submodule update --init --recursive

# build
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
```