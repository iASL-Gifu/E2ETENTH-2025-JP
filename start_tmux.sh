#!/bin/bash

SESSION_NAME="ros"
WINDOW_NAME="main"
ROS_WS_PATH="$HOME/E2ETENTH-2025-JP/ros2_ws" # ROS 2ワークスペースのパス (環境に合わせて変更してください)
SETUP_SCRIPT="source install/setup.bash"

# セッションが既に存在するか確認
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # セッションが存在しない場合、新しいセッションを作成 (ウィンドウ名もここで指定)
  # 最初のペインがペイン0として作成される
  tmux new-session -d -s $SESSION_NAME -n $WINDOW_NAME

  # ペイン0を縦に分割 (ペイン0が上、ペイン1が下になる)
  tmux split-window -v -t $SESSION_NAME:$WINDOW_NAME.0

  # 上のペイン (ペイン0) を横に分割 (ペイン0が左上、新しいペインが右上になり、インデックスは2になる)
  tmux split-window -h -t $SESSION_NAME:$WINDOW_NAME.0

  # 元の下のペイン (ペイン1) を選択する
  tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.1
  # 選択した下のペイン (ペイン1) を横に分割 (ペイン1が左下、新しいペインが右下になり、インデックスは3になる)
  tmux split-window -h # カレントペインが対象なので -t は省略可能

  # 各ペインでコマンドを実行
  # 左上 (ペイン0)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.0 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m
  
  # 右上 (ペイン2)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.2 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m

  # 左下 (ペイン1)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.1 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m
  
  # 右下 (ペイン3)
  tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.3 "cd $ROS_WS_PATH && $SETUP_SCRIPT && clear" C-m
  
  # 最後にアクティブにするペインを選択 (例: 左上のペイン0)
  tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.0

fi

# セッションにアタッチ
tmux attach-session -t $SESSION_NAME
