# iASL-E2ETENTH-2025

## setup dependencies
```bash
# apt から install
sudo apt install ros-${ROS_DISTRO}-laser-filters # for 2d lidar filter
sudo apt install ros-${ROS_DISTRO}-rosbag2-storage-mcap # for rosbag
```

## clone and build
```bash
# clone
cd ~
git clone https://github.com/iASL-Gifu/E2ETENTH-2025-JP.git

## .repos内のパッケージをclone
cd ~/E2ETENTH-2025-JP
vcs import src < packages.repos

## urg nodeのサブモジュールをclone
cd src/sensors/urg_node2/
git submodule update --init --recursive

# build
cd ~/E2ETENTH-2025-JP
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
```