#!/bin/bash

#=================================================
# ★ 設定項目 ★
#=================================================
# --- ローカルマシン側の設定 ---
# ★ 変更点 ★: コマンドライン引数からデータセットのパスを取得
if [ -z "$1" ]; then
    echo "エラー: データセットのパスが指定されていません。"
    echo "使用法: $0 <データセットのパス>"
    exit 1
fi
if [ ! -d "$1" ]; then
    echo "エラー: 指定されたパス '$1' はディレクトリとして存在しません。"
    exit 1
fi
DATA_PATH=$1


# --- ↓↓↓ ここから追加・変更 ↓↓↓ ---

# DATA_PATHから実験名を自動的に抽出 (例: 6-27-obstacle1-at)
TMP_DATA_PATH=${DATA_PATH%/} # 末尾にスラッシュがあれば削除
EXPERIMENT_NAME=$(basename ${TMP_DATA_PATH})

# 抽出した実験名を使って、チェックポイントと設定ファイルのパスを動的に生成
BASE_CKPT_PATH="./ckpts/${EXPERIMENT_NAME}/"
CONFIG_SAVE_PATH="./configs/${EXPERIMENT_NAME}/"

# --- ↑↑↑ ここまで追加・変更 ↑↑↑ ---

# --- Jetsonへの転送設定 ---
SSH_USER="tamiya"
SSH_HOST="10.42.0.1"
# Jetson側の転送先ディレクトリのベースパス
REMOTE_BASE_PATH="/home/tamiya/E2ETENTH-2025-JP/AT_repos/"


#=================================================
# モデル学習のパラメータ設定
#=================================================
# --- 時系列モデル用の設定 ---
BATCH_SIZE_RECURRENT=8
SEQ_LENGTH_RECURRENT=16
RANDOM_RATIOS_RECURRENT=(0.75)

# --- 非時系列モデル用の設定 ---
BATCH_SIZE_DEFAULT=64
RANDOM_RATIO_DEFAULT=1.0
SEQ_LENGTH_DEFAULT=1 # ★ 変更点 ★: 基本のsequence_lengthは1

# 実行するモデル名のリスト
MODELS=(
    "TinyLidarNet"
    "TinyLidarConvTransformerNet"
)


#=================================================
# 事前準備
#=================================================
echo "Creating local directories for checkpoints and configs..."
mkdir -p ${BASE_CKPT_PATH}
mkdir -p ${CONFIG_SAVE_PATH}
echo "Done."
echo ""

#=================================================
# 学習と設定ファイル生成の実行ループ
#=================================================
echo "Starting training and config generation for all models..."
echo "================================================================="
echo ""

for model_name in "${MODELS[@]}"; do

    # 時系列/非時系列モデルで学習パラメータを切り替え
    if [[ "$model_name" == *Lstm* || "$model_name" == *Transformer* ]]; then
        # --- 時系列モデルの場合 ---
        echo ">>> Processing Recurrent Model: ${model_name}"
        for random_ratio in "${RANDOM_RATIOS_RECURRENT[@]}"; do
            ckpt_path="${BASE_CKPT_PATH}${model_name}/ratio_${random_ratio}/"
            echo "  -> Running training with random_ratio: ${random_ratio}"

            # 学習コマンドの実行
            python3 train_cnn.py \
                data_path=${DATA_PATH} \
                ckpt_path=${ckpt_path} \
                model_name=${model_name} \
                batch_size=${BATCH_SIZE_RECURRENT} \
                sequence_length=${SEQ_LENGTH_RECURRENT} \
                random_ratio=${random_ratio}

            # ベストな重みファイルを自動で探し出す
            echo "  -> Finding the best model checkpoint..."
            best_model_filename=$(ls -1 "${ckpt_path}"model_epoch_*.pth | \
                sed -E 's/.*epoch_([0-9]+)_loss_([0-9.]+)\.pth/\2 \1 &/' | \
                sort -k1,1g -k2,2nr | \
                head -n 1 | \
                sed 's/.* //;s/.*\///')
            echo "     Best model found: ${best_model_filename}"

            # モデルに応じてsequence_lengthを決定
            if [[ "$model_name" == *Transformer* ]]; then
                current_seq_length=${SEQ_LENGTH_RECURRENT}
            else
                current_seq_length=${SEQ_LENGTH_DEFAULT}
            fi

            # 設定ファイルの生成
            CONFIG_FILE_PATH="${CONFIG_SAVE_PATH}config_${model_name}_ratio_${random_ratio}.yaml"
            echo "  -> Generating ROS2 config file: ${CONFIG_FILE_PATH}"
            cat <<EOF > ${CONFIG_FILE_PATH}
/**:
  ros__parameters:
    model_name: ${model_name}
    # ★ 変更点 ★: pthファイルまで含めたパスを記述 (EXPERIMENT_NAME を追加)
    model_path: ${REMOTE_BASE_PATH}ckpts/${EXPERIMENT_NAME}/${model_name}/ratio_${random_ratio}/${best_model_filename}
    sequence_length: ${current_seq_length}
    max_range: 30.0
    input_dim: 1081
    output_dim: 2
EOF
            echo "  -> Config file generated."
            echo ""
        done
    else
        # --- 非時系列モデルの場合 ---
        echo ">>> Processing Non-Recurrent Model: ${model_name}"
        ckpt_path="${BASE_CKPT_PATH}${model_name}/"
        echo "  -> Running training..."

        # 学習コマンドの実行
        python3 train_cnn.py \
            data_path=${DATA_PATH} \
            ckpt_path=${ckpt_path} \
            model_name=${model_name} \
            batch_size=${BATCH_SIZE_DEFAULT} \
            sequence_length=${SEQ_LENGTH_DEFAULT} \
            random_ratio=${RANDOM_RATIO_DEFAULT}

        # ベストな重みファイルを自動で探し出す
        echo "  -> Finding the best model checkpoint..."
        best_model_filename=$(ls -1 "${ckpt_path}"model_epoch_*.pth | \
            sed -E 's/.*epoch_([0-9]+)_loss_([0-9.]+)\.pth/\2 \1 &/' | \
            sort -k1,1g -k2,2nr | \
            head -n 1 | \
            sed 's/.* //;s/.*\///')
        echo "     Best model found: ${best_model_filename}"

        # 設定ファイルの生成
        CONFIG_FILE_PATH="${CONFIG_SAVE_PATH}config_${model_name}.yaml"
        echo "  -> Generating ROS2 config file: ${CONFIG_FILE_PATH}"
        cat <<EOF > ${CONFIG_FILE_PATH}
/**:
  ros__parameters:
    model_name: ${model_name}
    # ★ 変更点 ★: pthファイルまで含めたパスを記述 (EXPERIMENT_NAME を追加)
    model_path: ${REMOTE_BASE_PATH}ckpts/${EXPERIMENT_NAME}/${model_name}/${best_model_filename}
    sequence_length: ${SEQ_LENGTH_DEFAULT}
    max_range: 30.0
    input_dim: 1081
    output_dim: 2
EOF
        echo "  -> Config file generated."
        echo ""
    fi
done

echo "================================================================="
echo "All training and config generation processes have been completed."
echo ""


#=================================================
# ★ Jetsonへのファイル転送 ★
#=================================================
echo "Starting file transfer to Jetson [${SSH_USER}@${SSH_HOST}]..."
echo "================================================================="

# --- 1. 重みファイル(ckpts)の転送 ---
REMOTE_CKPT_PATH="${REMOTE_BASE_PATH}ckpts/"
echo "Transferring checkpoints to: ${REMOTE_CKPT_PATH}"
# ★ 変更点 ★: ディレクトリごと転送
scp -r ${BASE_CKPT_PATH} ${SSH_USER}@${SSH_HOST}:${REMOTE_CKPT_PATH}
echo " -> Checkpoints transfer complete."
echo ""

# --- 2. 設定ファイル(configs)の転送 ---
REMOTE_CONFIG_PATH="${REMOTE_BASE_PATH}configs/"
echo "Transferring config files to: ${REMOTE_CONFIG_PATH}"
# ★ 変更点 ★: ディレクトリごと転送
scp -r ${CONFIG_SAVE_PATH} ${SSH_USER}@${SSH_HOST}:${REMOTE_CONFIG_PATH}
echo " -> Config files transfer complete."
echo ""


echo "================================================================="
echo "All processes are finished successfully!"