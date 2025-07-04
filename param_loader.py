import os
import glob
import subprocess
import sys
import time
import argparse
import questionary
from questionary import Choice

def clear_screen():
    """画面をクリアする"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(command):
    """subprocessでコマンドを実行し、結果を返す"""
    return subprocess.run(command, capture_output=True, text=True, check=False)

def select_node():
    """ROS2ノードを選択する画面を表示し、選択されたノード名を返す"""
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
                    Choice("戻る", value="back")
                ]
            ).ask()
            if action == "back" or action is None:
                return None
            continue

        node_choices = [Choice(f"[{i+1}] {node}", value=node) for i, node in enumerate(nodes)]
        choices = node_choices + [
            questionary.Separator(),
            Choice("リストを更新", value="reload"),
            Choice("メインメニューに戻る", value="back")
        ]
        
        selected = questionary.select(
            "パラメータをロードするノードを選択してください:",
            choices=choices
        ).ask()

        if selected is None or selected == "back":
            return None
        if selected == "reload":
            continue
        
        return selected

def select_directory():
    """対話的にディレクトリを選択し、そのパスを返す"""
    current_path = os.getcwd()
    while True:
        clear_screen()
        print(f"📂 パラメータディレクトリを選択してください (現在のパス: {current_path})\n")
        
        try:
            items = sorted(os.listdir(current_path))
        except OSError as e:
            print(f"エラー: {e}")
            current_path = os.path.dirname(current_path)
            time.sleep(2)
            continue
        
        dir_items = [item for item in items if os.path.isdir(os.path.join(current_path, item))]
        
        item_choices = [Choice(f"[{i+1}] [{item}]/", value=item) for i, item in enumerate(dir_items)]
        
        choices = [
            Choice("✅ [ このディレクトリを決定する ]", value="."),
            Choice("⏪ ../ (親ディレクトリへ)", value=".."),
            questionary.Separator('---------- ディレクトリ一覧 ----------'),
        ] + item_choices + [
            questionary.Separator(),
            Choice("メインメニューに戻る", value="cancel")
        ]

        selected = questionary.select(
            "移動するディレクトリを選択 or このディレクトリを決定:",
            choices=choices
        ).ask()

        if selected is None or selected == "cancel":
            return None
        elif selected == ".":
            return current_path
        elif selected == "..":
            current_path = os.path.dirname(current_path)
        else:
            current_path = os.path.join(current_path, selected)

def select_and_load_param(node_name, param_dir):
    """YAMLファイルを選択してパラメータをロードする"""
    clear_screen()
    print(f"🎯 現在のノード: {node_name}")
    print(f"📂 対象ディレクトリ: {param_dir}\n")

    yaml_files = glob.glob(os.path.join(param_dir, '*.yaml'))
    yaml_files += glob.glob(os.path.join(param_dir, '*.yml'))

    if not yaml_files:
        print(f"❌ ディレクトリ '{param_dir}' にYAMLファイルが見つかりません。")
        questionary.text("Enterキーを押してメインメニューに戻ります...").ask()
        return

    file_choices = [
        Choice(f"[{i+1}] {os.path.basename(f)}", value=os.path.basename(f)) 
        for i, f in enumerate(sorted(yaml_files))
    ]
    
    choices = [
        Choice("[ 戻る ]", value="back"),
        questionary.Separator('--- YAML ファイル一覧 ---')
    ] + file_choices

    selected_yaml_name = questionary.select(
        "ロードするYAMLファイルを選択してください:",
        choices=choices,
    ).ask()
    
    if selected_yaml_name is None or selected_yaml_name == "back":
        return

    selected_yaml_path = os.path.join(param_dir, selected_yaml_name)
    print(f"\n⏳ 実行中: ros2 param load {node_name} {selected_yaml_path}")
    result = run_command(['ros2', 'param', 'load', node_name, selected_yaml_path])
    
    if result.returncode == 0:
        print("\n✅ パラメータのロードに成功しました。")
    else:
        print("\n❌ パラメータのロードに失敗しました。")
        print("--- エラー出力 ---\n" + result.stderr + "--------------------")
        
    questionary.text("Enterキーを押してメインメニューに戻ります...").ask()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='ROS2パラメータを対話的にロードするCUIツール'
    )
    parser.add_argument(
        'param_dir', 
        nargs='?',
        default=None,
        help='(任意) パラメータYAMLファイルが格納されているディレクトリパス'
    )
    args = parser.parse_args()
    
    # --- 状態管理 ---
    current_node = None
    current_dir = args.param_dir

    if current_dir and not os.path.isdir(current_dir):
        print(f"❌ エラー: 指定されたディレクトリ '{current_dir}' が見つかりません。")
        sys.exit(1)

    try:
        while True:
            clear_screen()
            
            # --- 現在の状態を表示 ---
            node_display = f"🎯 現在のノード: {current_node}" if current_node else "🎯 現在のノード: 🚫 未選択"
            dir_display = f"📂 現在のディレクトリ: {current_dir}" if current_dir else "📂 現在のディレクトリ: 🚫 未選択"
            print(f"ROS2 Param Loader\n{'-'*30}")
            print(node_display)
            print(dir_display)
            print(f"{'-'*30}\n")

            # --- アクション選択 ---
            main_choices = [
                Choice("ノードを選択/変更する", value="select_node"),
                Choice("ディレクトリを選択/変更する", value="select_dir"),
                questionary.Separator(),
            ]
            
            # ノードとディレクトリが両方選択されている場合のみロード選択肢を表示
            if current_node and current_dir:
                main_choices.append(Choice("YAMLファイルを選択してパラメータをロードする", value="load_param"))
            else:
                 main_choices.append(Choice("YAMLファイルをロード (ノードとディレクトリを選択してください)", value="load_param", disabled=True))

            main_choices.extend([
                questionary.Separator(),
                Choice("終了", value="exit")
            ])
            
            action = questionary.select(
                "実行するアクションを選択してください:",
                choices=main_choices
            ).ask()

            if action is None or action == "exit":
                break

            if action == "select_node":
                selected = select_node()
                if selected: # Noneでない場合（正常に選択された場合）のみ更新
                    current_node = selected
            
            elif action == "select_dir":
                selected = select_directory()
                if selected: # Noneでない場合のみ更新
                    current_dir = selected

            elif action == "load_param":
                if current_node and current_dir:
                    select_and_load_param(current_node, current_dir)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        clear_screen()
        print("👋 ツールを終了します。")

if __name__ == '__main__':
    main()