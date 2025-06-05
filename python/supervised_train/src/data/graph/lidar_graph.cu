#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h> 

#define DEG2RAD 0.017453292519943295f  // M_PI / 180.0

namespace py = pybind11;

// +++ 変更点 +++
// 近傍候補の距離とインデックスを保持するための構造体
struct Neighbor {
    float dist2; // 距離の2乗
    int index;   // ノードのインデックス
};


__global__ void polar_to_cartesian_kernel(
    const float* __restrict__ ranges,
    float* __restrict__ x,
    float* __restrict__ y,
    int num_points,
    float angle_increment
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float angle_deg = -135.0f + idx * angle_increment;
    float angle_rad = angle_deg * DEG2RAD;
    float r = ranges[idx];
    x[idx] = r * cosf(angle_rad);
    y[idx] = r * sinf(angle_rad);
}

// +++ 変更点: build_graph_kernel の内部ロジックを全面的に修正 +++
__global__ void build_graph_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int* __restrict__ edge_index,
    float* __restrict__ edge_attr,
    int num_points,
    float threshold2,
    int max_neighbors 
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    // 1. 各スレッドが近傍候補を保持するためのローカル配列を宣言
    //    ※このサイズは、1つの点が持ちうる近傍候補の最大数より大きくしてください。
    constexpr int MAX_CANDIDATES = 128;
    Neighbor candidates[MAX_CANDIDATES];
    int candidate_count = 0;

    // 2. 距離がしきい値以下の全ての近傍候補を見つけてローカル配列に格納
    for (int j = 0; j < num_points; j++) {
        if (i == j) continue;

        float dx = x[j] - x[i];
        float dy = y[j] - y[i];
        float d2 = dx * dx + dy * dy;

        if (d2 <= threshold2) {
            if (candidate_count < MAX_CANDIDATES) { // バッファオーバーフロー防止
                candidates[candidate_count].dist2 = d2;
                candidates[candidate_count].index = j;
                candidate_count++;
            }
        }
    }

    // 3. 見つけた近傍候補を距離でソート（単純な挿入ソート）
    for (int k = 1; k < candidate_count; k++) {
        Neighbor key = candidates[k];
        int l = k - 1;
        while (l >= 0 && candidates[l].dist2 > key.dist2) {
            candidates[l + 1] = candidates[l];
            l = l - 1;
        }
        candidates[l + 1] = key;
    }

    // 4. ソートされたリストから上位 `max_neighbors` 個をグローバルメモリに書き込む
    int num_edges_to_write = (candidate_count < max_neighbors) ? candidate_count : max_neighbors;
    for (int k = 0; k < num_edges_to_write; k++) {
        int write_idx = i * max_neighbors + k;
        edge_index[write_idx * 2 + 0] = i;
        edge_index[write_idx * 2 + 1] = candidates[k].index;
        edge_attr[write_idx] = sqrtf(candidates[k].dist2);
    }
}


std::vector<torch::Tensor> build_lidar_graph_cuda(
    torch::Tensor ranges, 
    float distance_threshold, 
    int max_neighbors
) {
    const int num_points = ranges.size(0);
    const float angle_increment = 270.0f / num_points;
    const float threshold2 = distance_threshold * distance_threshold;

    auto x = torch::zeros({num_points}, torch::device(ranges.device()).dtype(torch::kFloat32));
    auto y = torch::zeros({num_points}, torch::device(ranges.device()).dtype(torch::kFloat32));

    polar_to_cartesian_kernel<<<(num_points + 255) / 256, 256>>>(
        ranges.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(),
        num_points, angle_increment
    );

    auto edge_index = torch::full({num_points * max_neighbors, 2}, -1, torch::dtype(torch::kInt32).device(ranges.device()));
    auto edge_attr = torch::zeros({num_points * max_neighbors}, torch::dtype(torch::kFloat32).device(ranges.device()));

    build_graph_kernel<<<(num_points + 255) / 256, 256>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        edge_index.data_ptr<int>(), edge_attr.data_ptr<float>(),
        num_points, threshold2,
        max_neighbors 
    );

    return {edge_index, edge_attr, x, y};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_lidar_graph_cuda", &build_lidar_graph_cuda, "Build LiDAR Graph (CUDA)",
          py::arg("ranges"),
          py::arg("distance_threshold"),
          py::arg("max_neighbors") = 5
    );
}