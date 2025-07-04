# Supervised Train

このプロジェクトは、教師あり学習でE2Eモデルを学習させることを目的としています。
## setup


###
Pythonのスクリプトを動作させるときは、仮想環境を作成することをお勧めします。venvを使って動作済みです。
```bash
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

# for gnn
cd python/lidar_graph/
python3 setup.py build_ext --inplace
```

## train on simulator dataset

### 1. collect data from F1tenth-gym 
```bash
## using pure pursuit controller 
python3 collect_data_sim.py \
num_sets=1 \
output_dir=./datasets/sim/<run_name>
```

### 2. train on collected sim dataset
hydraを使って、実行時にパラメータを変えることができます。詳細なパラメータについては、 [yamlファイル](./config/train_cnn.yaml)内部を参照してください。
```bash
python3 train_cnn.py \
model_name=<model> \
data_path=./datasets/ \
ckpt_path=./ckpts/ \
sequence_length=<length>  \
random_ratio=0.5 \
batch_size=8
```

### 3. benchmark on simulator
テストマップにて学習の評価を行います。
```bash
python3 benchmark_sim.py \
ckpt_path=./ckpts/
```

## train on rosbag data

### 1. collect data from real environment
ros2 ノードを起動する毎に時刻に応じたディレクトリが作成されます
```bash
source ~/E2E_TENETH/ros2_ws/install/setup.bash
ros2 launch bag_manager_py bag_manager_node
```

### 2. Data Processing
ros2のbagからtopicを抽出し、npyに保存する。
*注意 これは作成した仮想環境ではなく、ローカルのros2が利用可能な環境で実行する必要があります。
```bash
/path/to/rosbag_dir/  
├── bag_1/
│   ├── metadata.yaml
│   └── bag_1_0.db3  
│
├── bag_2/
│   ├── metadata.yaml
│   └── bag_2_0.db3
│
└── bag_3/
    ├── metadata.yaml
    └── bag_3_0.db3
    ...
```

```bash
python3 extract_topics.py \
--bags_dir path/to/rosbag_dir \
--outdir path/to/outdir\
--scan_topic \
--cmd_topic\
```

### 3. train on rosbag dataset
config/train_cnn.yamlを編集し、実行します。
*注意　これは仮想環境で実行します。
```bash
python3 train_cnn.py \
model_name=<model> \
data_path=./datasets/ \
ckpt_path=./ckpts/ \
sequence_length=<length> \
random_ratio=0.5 \
batch_size=8
```

### 3. benchmark on simulator
シミュレータのテストマップにて学習の評価を行います。
```bash
python3 benchmark_sim.py \
ckpt_path=./ckpts/



