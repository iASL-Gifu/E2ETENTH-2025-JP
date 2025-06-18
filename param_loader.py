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

        # Choiceオブジェクトを使い、表示名(title)と内部値(value)を分ける
        numbered_node_choices = []
        for i, node in enumerate(nodes, 1):
            numbered_node_choices.append(
                Choice(title=f"[{i}] {node}", value=node)
            )

        choices = numbered_node_choices + [
            questionary.Separator(),
            Choice("ノードリストを更新", value="reload"),
            Choice("終了", value="exit")
        ]
        
        # questionaryはvalue値を返すので、後続の処理は変更不要
        selected = questionary.select(
            "パラメータをロードするノードを選択してください:",
            choices=choices
        ).ask()

        if selected is None or selected == "exit":
            return None
        if selected == "reload":
            continue
        
        return selected

def select_directory_loop():
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
        
        numbered_item_choices = []
        # 抽出したディレクトリリストに対して番号を振る
        for i, item in enumerate(dir_items, 1):
            display_name = f"[{i}] [{item}]/"
            numbered_item_choices.append(Choice(title=display_name, value=item))
        
        choices = [
            Choice("✅ [ このディレクトリを決定する ]", value="."),
            Choice("⏪../", value=".."),
            questionary.Separator('---------- ディレクトリ一覧 ----------') # ファイルがなくなったので名称変更
        ] + numbered_item_choices

        selected = questionary.select(
            "移動するディレクトリを選択 or このディレクトリを決定 (矢印キーで選択):",
            choices=choices
        ).ask()

        if selected is None: # Ctrl+C
            return None
        
        elif selected == ".":
            return current_path
        elif selected == "..":
            current_path = os.path.dirname(current_path)
        
        
        else:
            current_path = os.path.join(current_path, selected)

def param_load_loop(node_name, param_dir):
    """指定されたノードに対して、YAMLの選択とロードを繰り返すループ"""
    while True:
        clear_screen()
        print(f"✅ 現在の選択ノード: {node_name}")
        print(f"📂 対象ディレクトリ: {param_dir}\n")
        
        action = questionary.select(
            "実行するアクションを選択してください:",
            choices=[
                Choice("YAMLファイルを選択してロード", value="load"),
                Choice("ノードを再選択する", value="reselect_node"),
                Choice("終了", value="exit")
            ]
        ).ask()

        if action is None or action == "exit":
            raise SystemExit()

        if action == "reselect_node":
            return

        if action == "load":
            while True:
                yaml_files = glob.glob(os.path.join(param_dir, '*.yaml'))
                yaml_files += glob.glob(os.path.join(param_dir, '*.yml'))

                if not yaml_files:
                    print(f"❌ ディレクトリ '{param_dir}' にYAMLファイルが見つかりません。")
                    retry_action = questionary.select(
                        "どうしますか？",
                        choices=[Choice("リトライ", value="retry"), Choice("アクション選択に戻る", value="back")]
                    ).ask()
                    if retry_action == "back" or retry_action is None: break
                    else:
                        clear_screen()
                        print(f"✅ 現在の選択ノード: {node_name}\n")
                        continue
                
                numbered_file_choices = []
                for i, filename in enumerate(sorted([os.path.basename(f) for f in yaml_files]), 1):
                    numbered_file_choices.append(
                        Choice(title=f"[{i}] {filename}", value=filename)
                    )

                choices = [
                    Choice("[ 戻る ]", value="back"),
                    questionary.Separator('--- YAML ファイル一覧 ---')
                ] + numbered_file_choices

                selected_yaml_name = questionary.select(
                    "ロードするYAMLファイルを選択してください (矢印キーで選択):",
                    choices=choices,
                ).ask()
                
                if selected_yaml_name is None or selected_yaml_name == "back":
                    break

                selected_yaml_path = os.path.join(param_dir, selected_yaml_name)
                print(f"\n⏳ 実行中: ros2 param load {node_name} {selected_yaml_path}")
                result = run_command(['ros2', 'param', 'load', node_name, selected_yaml_path])
                if result.returncode == 0:
                    print("\n✅ パラメータのロードに成功しました。")
                else:
                    print("\n❌ パラメータのロードに失敗しました。")
                    print("--- エラー出力 ---\n" + result.stderr + "--------------------")
                questionary.text("Enterキーを押して続行...").ask()
                break

# ★ main関数を修正
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='A CUI tool to load ROS2 parameters interactively.'
    )
    # ★引数を必須(required)から任意(optional)に変更
    parser.add_argument(
        'param_dir', 
        nargs='?', # 0か1個の引数を受け取る
        default=None, # 引数がなければNoneになる
        help='(Optional) Directory path where YAML parameter files are stored.'
    )
    args = parser.parse_args()
    
    # --- 引数の有無で動作を分岐 ---
    param_dir_from_arg = args.param_dir
    
    try:
        if param_dir_from_arg:
            # ケースA: 引数が指定された場合
            if not os.path.isdir(param_dir_from_arg):
                print(f"❌ Error: Directory not found at '{param_dir_from_arg}'")
                sys.exit(1)
            
            while True:
                selected_node = select_node_loop()
                if selected_node is None: break
                param_load_loop(selected_node, param_dir_from_arg)
        else:
            # ケースB: 引数が指定されなかった場合
            while True:
                selected_node = select_node_loop()
                if selected_node is None: break
                
                # ディレクトリ選択モードを開始
                selected_dir = select_directory_loop()
                if selected_dir is None: continue # ディレクトリ選択をキャンセルしたらノード選択に戻る
                
                param_load_loop(selected_node, selected_dir)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        clear_screen()
        print("👋 ツールを終了します。")

if __name__ == '__main__':
    main()