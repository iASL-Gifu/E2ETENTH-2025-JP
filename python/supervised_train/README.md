# Supervised Train

このプロジェクトは、教師あり学習でE2Eモデルを学習させることを目的としています。

## setup

```bash
python3.11 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

# for gnn
python3 setup.py build_ext --inplace
```

## Data Processing
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
## train
config/train_cnn.yamlを編集し、実行します。
*注意　これは仮想環境で実行します。
```bash
python3 train_cnn.py \
model_name=<model> \
data_path=./datasets/ \
ckpt_path=./ckpts/ \
sequence_length=<length> \
random_ratio=0.5 \

```