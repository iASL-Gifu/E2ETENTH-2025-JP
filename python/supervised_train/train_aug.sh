#!/bin/bash

#=================================================
# ★ 設定項目 ★
#=================================================
# --- ローカルマシン側の設定 ---
# コマンドライン引数からデータセットのパスを取得
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

# データパスから実験名を自動抽出し、各種パスを動的に生成
TMP_DATA_PATH=${DATA_PATH%/} # 末尾のスラッシュを削除
EXPERIMENT_NAME=$(basename ${TMP_DATA_PATH})
BASE_CKPT_PATH="./ckpts/${EXPERIMENT_NAME}/"
CONFIG_SAVE_PATH="./configs/${EXPERIMENT_NAME}/" # ★改良: 設定ファイルも実験ごとにフォルダ分け

# --- Jetsonへの転送設定 ---
SSH_USER="tamiya"
SSH_HOST="10.42.0.1"
REMOTE_BASE_PATH="/home/tamiya/E2ETENTH-2025-JP/AT_repos/"


#=================================================
# モデル学習のパラメータ設定
#=================================================
# --- 時系列モデル用の設定 ---
BATCH_SIZE_RECURRENT=8
SEQ_LENGTH_RECURRENT=16
RANDOM_RATIOS_RECURRENT=(0.8)

# --- 非時系列モデル用の設定 ---
BATCH_SIZE_DEFAULT=64
RANDOM_RATIO_DEFAULT=1.0
SEQ_LENGTH_DEFAULT=1

# 実行するモデル名のリスト (必要に応じて時系列モデルも追加可能)
MODELS=(
    "TinyLidarNet"
    # "TinyLidarConvTransformerNet"
)

#=================================================
# 事前準備
#=================================================
echo "Creating local directories for experiment: ${EXPERIMENT_NAME}"
mkdir -p ${BASE_CKPT_PATH}
mkdir -p ${CONFIG_SAVE_PATH}
echo "  -> Checkpoints: ${BASE_CKPT_PATH}"
echo "  -> Configs:     ${CONFIG_SAVE_PATH}"
echo "Done."
echo ""

#=================================================
# 学習と設定ファイル生成の実行ループ
#=================================================
echo "Starting training and config generation for all models..."
echo "================================================================="
echo ""

for model_name in "${MODELS[@]}"; do

    # モデル名に "Lstm" または "Transformer" が含まれているか判定
    if [[ "$model_name" == *Lstm* || "$model_name" == *Transformer* ]]; then
        # --- 時系列モデルの場合の処理 (最初のスクリプトのロジックを流用) ---
        echo ">>> Processing Recurrent Model: ${model_name}"
        for random_ratio in "${RANDOM_RATIOS_RECURRENT[@]}"; do
            ckpt_path="${BASE_CKPT_PATH}${model_name}/ratio_${random_ratio}/"
            echo "  -> Running training with random_ratio: ${random_ratio}"

            python3 train_cnn.py \
                data_path=${DATA_PATH} ckpt_path=${ckpt_path} model_name=${model_name} \
                batch_size=${BATCH_SIZE_RECURRENT} sequence_length=${SEQ_LENGTH_RECURRENT} random_ratio=${random_ratio}

            echo "  -> Finding the best model checkpoint..."
            best_model_filename=$(ls -1 "${ckpt_path}"model_epoch_*.pth | \
                sed -E 's/.*epoch_([0-9]+)_loss_([0-9.]+)\.pth/\2 \1 &/' | sort -k1,1g -k2,2nr | head -n 1 | sed 's/.* //;s/.*\///')
            echo "     Best model found: ${best_model_filename}"

            CONFIG_FILE_PATH="${CONFIG_SAVE_PATH}config_${model_name}_ratio_${random_ratio}.yaml"
            echo "  -> Generating ROS2 config file: ${CONFIG_FILE_PATH}"
            cat <<EOF > ${CONFIG_FILE_PATH}
/**:
  ros__parameters:
    model_name: ${model_name}
    model_path: ${REMOTE_BASE_PATH}ckpts/${EXPERIMENT_NAME}/${model_name}/ratio_${random_ratio}/${best_model_filename}
    sequence_length: ${SEQ_LENGTH_RECURRENT}
    max_range: 30.0
    input_dim: 1081
    output_dim: 2
EOF
            echo "  -> Config file generated."
            echo ""
        done

    else
        # --- 非時系列モデルの場合の処理 (データ拡張比較) ---
        echo ">>> Processing Non-Recurrent Model for Augmentation Comparison: ${model_name}"
        echo "--------------------------------------------------------------------"
        AUG_PATTERNS=("with_aug" "no_aug")

        for pattern in "${AUG_PATTERNS[@]}"; do
            if [[ "$pattern" == "with_aug" ]]; then
                flip_prob_val=0.5; noise_std_val=0.01
                echo "  -> Running with augmentation (flip & noise)"
            else
                flip_prob_val=0.0; noise_std_val=0.0
                echo "  -> Running without augmentation"
            fi
            
            ckpt_path="${BASE_CKPT_PATH}${model_name}/${pattern}/"
            echo "     Executing training for pattern: [${pattern}]"
            python3 train_cnn.py \
                data_path=${DATA_PATH} ckpt_path=${ckpt_path} model_name=${model_name} \
                batch_size=${BATCH_SIZE_DEFAULT} sequence_length=${SEQ_LENGTH_DEFAULT} random_ratio=${RANDOM_RATIO_DEFAULT} \
                flip_prob=${flip_prob_val} noise_std=${noise_std_val}

            echo "     Finding the best model checkpoint for [${pattern}]..."
            best_model_filename=$(ls -1 "${ckpt_path}"model_epoch_*.pth | \
                sed -E 's/.*epoch_([0-9]+)_loss_([0-9.]+)\.pth/\2 \1 &/' | sort -k1,1g -k2,2nr | head -n 1 | sed 's/.* //;s/.*\///')
            echo "     Best model found: ${best_model_filename}"

            CONFIG_FILE_PATH="${CONFIG_SAVE_PATH}config_${model_name}_${pattern}.yaml"
            echo "     Generating ROS2 config file: ${CONFIG_FILE_PATH}"
            cat <<EOF > ${CONFIG_FILE_PATH}
/**:
  ros__parameters:
    model_name: ${model_name}
    model_path: ${REMOTE_BASE_PATH}ckpts/${EXPERIMENT_NAME}/${model_name}/${pattern}/${best_model_filename}
    sequence_length: ${SEQ_LENGTH_DEFAULT}
    max_range: 30.0
    input_dim: 1081
    output_dim: 2
EOF
            echo "     Config file generated."
            echo ""
        done
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
# ディレクトリごと転送することで、Jetson側にも同じ構成を維持
scp -r ${BASE_CKPT_PATH} ${SSH_USER}@${SSH_HOST}:${REMOTE_CKPT_PATH}
echo " -> Checkpoints transfer complete."
echo ""

# --- 2. 設定ファイル(configs)の転送 ---
REMOTE_CONFIG_PATH="${REMOTE_BASE_PATH}configs/"
echo "Transferring config files to: ${REMOTE_CONFIG_PATH}"
scp -r ${CONFIG_SAVE_PATH} ${SSH_USER}@${SSH_HOST}:${REMOTE_CONFIG_PATH}
echo " -> Config files transfer complete."
echo ""


echo "================================================================="
echo "All processes are finished successfully!"