import lidar_graph, lidar_graph_cuda
from torch_geometric.data import Data, Batch
import torch

def build_batch_graph(scan_data_batch, distance_threshold=1.0, max_edges=5):
    """
    C実装されたSpiral風LiDARグラフ構築を用いて、LiDARスキャンデータをグラフに変換。
    
    Args:
        scan_data_batch (Tensor): [batch_size, num_points] のLiDARスキャンデータ
        angle_radius (int): 探索角度インデックス範囲（±N で探索）
        distance_threshold (float): エッジとして接続可能な最大距離

    Returns:
        Batch: torch_geometric.data.Batch オブジェクト（各バッチのグラフ）
    """
    batch_size, num_points = scan_data_batch.size()

    # C側の初期化（探索範囲・距離しきい値を指定）
    lidar_graph.initialize(num_points, batch_size, distance_threshold, max_edges)

    # Pythonリストに変換
    lidar_data = [scan_data_batch[i].tolist() for i in range(batch_size)]

    # グラフ構築（エッジリスト取得）
    edge_lists = lidar_graph.build_graph(lidar_data)

    # ノード位置（x, y）取得
    node_positions_batch = lidar_graph.get_node_positions()

    graphs = []

    for i, edge_list in enumerate(edge_lists):
        node_positions = node_positions_batch[i]
        x = torch.tensor(node_positions, dtype=torch.float32)

        if len(edge_list) > 0:
            edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([e[2] for e in edge_list], dtype=torch.float32).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    return Batch.from_data_list(graphs)


def build_batch_graph_cuda(scan_data_batch, distance_threshold=1.0, max_edges=5, use_cuda=True):
    """
    LiDARスキャンデータからグラフを構築（CUDA優先）。
    
    Args:
        scan_data_batch (Tensor): [B, N] の LiDARスキャン（float）
        distance_threshold (float): 接続する最大距離
        use_cuda (bool): TrueならCUDA実装を使う（環境によっては自動で無効）

    Returns:
        Batch: torch_geometric.data.Batch
    """
    batch_size, num_points = scan_data_batch.size()
    graphs = []

    for i in range(batch_size):
        scan = scan_data_batch[i]

        if use_cuda and scan.is_cuda:
            # CUDA版を呼び出す
            edge_index, edge_attr, x, y = lidar_graph_cuda.build_lidar_graph_cuda(scan, distance_threshold)

            # 無効なエッジを除去
            mask = edge_index[:, 0] >= 0
            edge_index = edge_index[mask].t().contiguous()
            edge_attr = edge_attr[mask].view(-1, 1)
            x = torch.stack([x, y], dim=1)

        else:
            # C版を使用
            lidar_graph.initialize(num_points, 1, distance_threshold)
            edge_list = lidar_graph.build_graph([scan.tolist()])[0]
            node_positions = lidar_graph.get_node_positions()[0]
            x = torch.tensor(node_positions, dtype=torch.float32)

            if len(edge_list) > 0:
                edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor([e[2] for e in edge_list], dtype=torch.float32).view(-1, 1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)

    return Batch.from_data_list(graphs)