import os
import glob
import subprocess
import sys
import time
import argparse
import questionary
from questionary import Choice

# --- スクリプト上部のPARAM_DIR定義は不要なため削除 ---

def clear_screen():
    """画面をクリアする"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(command):
    """subprocessでコマンドを実行し、結果を返す"""
    return subprocess.run(command, capture_output=True, text=True)

def select_node_loop():
    """ノードを選択するループ。選択されたノード名を返すか、Noneを返す"""
    while True:
        clear_screen()
        print("🔍 アクティブなROS2ノードを検索中...")
        result = run_command(['ros2', 'node', 'list'])
        nodes = [node for node in result.stdout.strip().split('\n') if node]

        if not nodes:
            print("❌ アクティブなノードが見つかりません。")
            action = questionary.select(
                "どうしますか？",
                choices=[
                    Choice("リトライ", value="retry"),
                    Choice("終了", value="exit")
                ]
            ).ask()
            if action == "exit" or action is None:
                return None
            time.sleep(1)
            continue

        choices = nodes + [
            questionary.Separator(),
            Choice("ノードリストを更新", value="reload"),
            Choice("終了", value="exit")
        ]
        
        selected = questionary.select(
            "パラメータをロードするノードを選択してください:",
            choices=choices
        ).ask()

        if selected is None or selected == "exit":
            return None
        if selected == "reload":
            continue
        
        return selected

# ★修正点1: param_dirを引数として受け取るように修正
def param_load_loop(node_name, param_dir):
    """指定されたノードに対して、YAMLの選択とロードを繰り返すループ"""
    while True:
        clear_screen()
        print(f"✅ 現在の選択ノード: {node_name}")
        print(f"📂 対象ディレクトリ: {param_dir}\n") # 引数のparam_dirを表示
        
        action = questionary.select(
            "実行するアクションを選択してください:",
            choices=[
                Choice("YAMLファイルを選択してロード", value="load"),
                Choice("ノードを再選択する", value="reselect_node"),
                # ★修正点2: "ealue" -> "value" にタイプミスを修正
                Choice("終了", value="exit")
            ]
        ).ask()

        if action is None or action == "exit":
            raise SystemExit()

        if action == "reselect_node":
            return

        if action == "load":
            while True:
                print("📂 YAMLファイルを検索中...")
                # ★修正点3: 引数のparam_dirを使ってファイルを検索
                yaml_files = glob.glob(os.path.join(param_dir, '*.yaml'))
                yaml_files += glob.glob(os.path.join(param_dir, '*.yml'))

                if not yaml_files:
                    # 引数のparam_dirを表示
                    print(f"❌ ディレクトリ '{param_dir}' にYAMLファイルが見つかりません。")
                    retry_action = questionary.select(
                        "どうしますか？",
                        choices=[
                            Choice("リトライ", value="retry"),
                            Choice("アクション選択に戻る", value="back")
                        ]
                    ).ask()
                    if retry_action == "back" or retry_action is None:
                        break
                    else:
                        clear_screen()
                        print(f"✅ 現在の選択ノード: {node_name}\n")
                        continue

                choices = [os.path.basename(f) for f in yaml_files]
                choices.append(questionary.Separator())
                choices.append(Choice("戻る", value="back"))

                selected_yaml_name = questionary.select(
                    "ロードするYAMLファイルを選択してください:",
                    choices=choices
                ).ask()
                
                if selected_yaml_name is None or selected_yaml_name == "back":
                    break

                # 引数のparam_dirを使ってフルパスを作成
                selected_yaml_path = os.path.join(param_dir, selected_yaml_name)
                
                print(f"\n⏳ 実行中: ros2 param load {node_name} {selected_yaml_path}")
                result = run_command(['ros2', 'param', 'load', node_name, selected_yaml_path])

                if result.returncode == 0:
                    print("\n✅ パラメータのロードに成功しました。")
                else:
                    print("\n❌ パラメータのロードに失敗しました。")
                    print("--- エラー出力 ---")
                    print(result.stderr)
                    print("--------------------")
                
                questionary.text("Enterキーを押して続行...").ask()
                break

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='A CUI tool to load ROS2 parameters interactively.'
    )
    parser.add_argument(
        'param_dir', 
        help='Directory path where YAML parameter files are stored.'
    )
    args = parser.parse_args()
    param_dir = args.param_dir

    if not os.path.isdir(param_dir):
        print(f"❌ Error: Directory not found at '{param_dir}'")
        sys.exit(1)

    try:
        while True:
            selected_node = select_node_loop()
            if selected_node is None:
                break
            # 修正されたparam_load_loopを呼び出す
            param_load_loop(selected_node, param_dir)
            
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        clear_screen()
        print("👋 ツールを終了します。")

if __name__ == '__main__':
    main()