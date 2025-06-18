#!/bin/bash

# --- 設定項目 ---
DEFAULT_SESSION_NAME="ros"                     # デフォルトのセッション名
WINDOW_NAME="main"                             # ウィンドウ名
ROS_WS_PATH="$HOME/E2ETENTH-2025-JP/ros2_ws" # ROS 2ワークスペースのパス (環境に合わせて変更してください)
SETUP_SCRIPT="source install/setup.bash"

# --- セッション名の決定 ---
# スクリプト実行時に引数が与えられていれば、それをセッション名として使用する
# 例: ./script.sh my_session  -> my_session という名前で作成
# 例: ./script.sh             -> ros (デフォルト名) という名前で作成
if [ -n "$1" ]; then
  SESSION_NAME="$1"
else
  SESSION_NAME="$DEFAULT_SESSION_NAME"
fi

# --- tmuxセッションの準備 ---
# セッションが既に存在するか確認
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # セッションが存在しない場合、新しいセッションとペインを作成・設定する

  # 1. 新しいセッションを作成 (この時点でペイン0が作成される)
  tmux new-session -d -s $SESSION_NAME -n $WINDOW_NAME

  # 2. ペイン0を縦に分割 (上がペイン0、下がペイン1になる)
  # -p オプションでペインのサイズ比率を指定できます (例: -p 66 で上のペインを66%に)
  tmux split-window -v -t $SESSION_NAME:$WINDOW_NAME.0

  # 3. 下のペイン (ペイン1) を横に分割 (左がペイン1、右がペイン2になる)
  tmux split-window -h -t $SESSION_NAME:$WINDOW_NAME.1

  # --- 各ペインでコマンドを実行 ---
  # 上 (ペイン0)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.0 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m
  
  # 左下 (ペイン1)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.1 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m

  # 右下 (ペイン2)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.2 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m
  
  # 最後にアクティブにするペインを選択 (例: 上のペイン0)
  tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.0

fi

# セッションにアタッチ
tmux attach-session -t $SESSION_NAME