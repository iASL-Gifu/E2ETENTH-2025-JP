#!/bin/bash

# --- 設定 ---
# ペアリングしたいDualShock 4のMACアドレス
CONTROLLER_MAC="A0:AB:51:5F:62:86"

# --- スクリプト本体 ---
echo "🎮 DualShock 4 自動ペアリングスクリプト"
echo "========================================"
echo

# 1. ユーザーへの指示
echo "コントローラーのPSボタンとSHAREボタンを長押しして、"
echo "ライトバーが【白く点滅】するペアリングモードにしてください。"
echo
read -p "準備ができたら Enterキー を押してください..."

# 2. bluetoothctlを非対話的に実行
echo
echo "ペアリングを開始します (MACアドレス: $CONTROLLER_MAC)"
echo "しばらくお待ちください..."

# コマンドをパイプでbluetoothctlに渡すことで自動実行する
# 各コマンドの間にsleepを挟み、処理の安定性を高める
{
    # 念のため、既存のペアリング情報を削除
    echo -e "remove $CONTROLLER_MAC\n"
    sleep 2
    
    # スキャンを開始してデバイスを見つける
    echo -e "scan on\n"
    # スキャン時間を10秒確保
    sleep 10
    echo -e "scan off\n"
    sleep 1
    
    # ペアリング、信頼、接続
    echo -e "pair $CONTROLLER_MAC\n"
    sleep 3
    echo -e "trust $CONTROLLER_MAC\n"
    sleep 2
    echo -e "connect $CONTROLLER_MAC\n"
    sleep 4

} | sudo bluetoothctl

echo
echo "========================================"
echo "✅ 処理が完了しました。"
echo "コントローラーのライトバーが【青色に点灯】していれば成功です。"
echo
echo "もし失敗した場合は、コントローラーを再度ペアリングモードにしてから"
echo "このスクリプトをもう一度実行してください。"