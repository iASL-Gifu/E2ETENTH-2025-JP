#include <Python.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI / 180.0)
#define MAX_NEIGHBORS 2048  // 一時的な候補バッファの上限

// 構造体定義
typedef struct {
    double x;
    double y;
} Node;

typedef struct {
    int from;
    int to;
    double weight;
} Edge;

typedef struct {
    double dist2;
    int index;
} Neighbor;

// グローバルデータ
static Node** nodes = NULL;
static Edge** edges = NULL;
static int* edge_counts = NULL;
static int num_nodes = 0;
static int batch_size = 0;
static double distance_threshold = 1.0;
static double threshold2 = 1.0;

static int max_edges_per_node = 5;

// qsort用の比較関数
int compare_neighbors(const void* a, const void* b) {
    double d1 = ((Neighbor*)a)->dist2;
    double d2 = ((Neighbor*)b)->dist2;
    return (d1 > d2) - (d1 < d2);  // 距離の昇順でソート
}

// モジュールの初期化とメモリ確保
static PyObject* initialize(PyObject* self, PyObject* args) {
    int new_num_nodes, new_batch_size;
    double new_distance_threshold;
    int new_max_edges = -1; // オプショナル引数のための初期値

    if (!PyArg_ParseTuple(args, "iid|i", &new_num_nodes, &new_batch_size, &new_distance_threshold, &new_max_edges)) {
        return NULL;
    }

    // 既存のメモリを解放
    if (nodes) {
        for (int i = 0; i < batch_size; i++) {
            free(nodes[i]);
            free(edges[i]);
        }
        free(nodes);
        free(edges);
        free(edge_counts);
    }

    // グローバル変数を更新
    num_nodes = new_num_nodes;
    batch_size = new_batch_size;
    distance_threshold = new_distance_threshold;
    threshold2 = distance_threshold * distance_threshold;
    if (new_max_edges != -1) {
        max_edges_per_node = new_max_edges;
    }

    // メモリを確保
    nodes = (Node**)malloc(batch_size * sizeof(Node*));
    edges = (Edge**)malloc(batch_size * sizeof(Edge*));
    edge_counts = (int*)malloc(batch_size * sizeof(int));

    for (int b = 0; b < batch_size; b++) {
        nodes[b] = (Node*)malloc(num_nodes * sizeof(Node));
        // 変更点: 可変の最大エッジ数に基づいてメモリを確保
        edges[b] = (Edge*)malloc(num_nodes * max_edges_per_node * sizeof(Edge));
        edge_counts[b] = 0;
    }

    Py_RETURN_NONE;
}

// グラフを構築するメイン関数
static PyObject* build_graph(PyObject* self, PyObject* args) {
    PyObject* input_lists;
    if (!PyArg_ParseTuple(args, "O", &input_lists)) {
        return NULL;
    }

    if (!PyList_Check(input_lists) || PyList_Size(input_lists) != batch_size) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list of batch_size lists");
        return NULL;
    }

    double angle_increment = 270.0 / num_nodes;

    for (int b = 0; b < batch_size; b++) {
        PyObject* scan = PyList_GetItem(input_lists, b);
        if (!PyList_Check(scan)) {
            PyErr_SetString(PyExc_TypeError, "Each scan must be a list");
            return NULL;
        }

        // ノードを極座標→直交座標に変換
        for (int i = 0; i < num_nodes; i++) {
            double r = PyFloat_AsDouble(PyList_GetItem(scan, i));
            double angle = (-135.0 + i * angle_increment) * DEG2RAD;
            nodes[b][i].x = r * cos(angle);
            nodes[b][i].y = r * sin(angle);
        }

        edge_counts[b] = 0;
        for (int i = 0; i < num_nodes; i++) {
            Neighbor neighbors[MAX_NEIGHBORS];
            int count = 0;

            // 距離条件を満たす近傍候補を探索
            for (int j = 0; j < num_nodes; j++) {
                if (i == j) continue;

                double dx = nodes[b][j].x - nodes[b][i].x;
                double dy = nodes[b][j].y - nodes[b][i].y;
                double dist2 = dx * dx + dy * dy;

                if (dist2 <= threshold2) {
                    // 変更点: バッファオーバーフローを防ぐための安全チェック
                    if (count < MAX_NEIGHBORS) {
                        neighbors[count].index = j;
                        neighbors[count].dist2 = dist2;
                        count++;
                    } else {
                        // バッファが満杯になったら、警告を出してこの点の探索を打ち切ることも可能
                        // PyErr_WarnEx(PyExc_Warning, "Neighbor buffer is full, some neighbors might be ignored.", 1);
                        break;
                    }
                }
            }

            // 近傍候補を距離が近い順に並べる
            qsort(neighbors, count, sizeof(Neighbor), compare_neighbors);

            for (int k = 0; k < max_edges_per_node && k < count; k++) {
                int eidx = edge_counts[b]++;
                edges[b][eidx].from = i;
                edges[b][eidx].to = neighbors[k].index;
                edges[b][eidx].weight = sqrt(neighbors[k].dist2);
            }
        }
    }

    // 結果をPythonのリストとして返す
    PyObject* batch_edge_list = PyList_New(batch_size);
    for (int b = 0; b < batch_size; b++) {
        PyObject* edge_list = PyList_New(edge_counts[b]);
        for (int i = 0; i < edge_counts[b]; i++) {
            PyObject* tpl = PyTuple_New(3);
            PyTuple_SetItem(tpl, 0, PyLong_FromLong(edges[b][i].from));
            PyTuple_SetItem(tpl, 1, PyLong_FromLong(edges[b][i].to));
            PyTuple_SetItem(tpl, 2, PyFloat_FromDouble(edges[b][i].weight));
            PyList_SetItem(edge_list, i, tpl);
        }
        PyList_SetItem(batch_edge_list, b, edge_list);
    }

    return batch_edge_list;
}

// ノードの座標を取得する関数
static PyObject* get_node_positions(PyObject* self, PyObject* args) {
    PyObject* batch_node_list = PyList_New(batch_size);
    for (int b = 0; b < batch_size; b++) {
        PyObject* node_list = PyList_New(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            PyObject* coords = PyTuple_New(2);
            PyTuple_SetItem(coords, 0, PyFloat_FromDouble(nodes[b][i].x));
            PyTuple_SetItem(coords, 1, PyFloat_FromDouble(nodes[b][i].y));
            PyList_SetItem(node_list, i, coords);
        }
        PyList_SetItem(batch_node_list, b, node_list);
    }
    return batch_node_list;
}

// Pythonモジュールに登録するメソッドの定義
static PyMethodDef LidarGraphMethods[] = {
    {"initialize", initialize, METH_VARARGS, "initialize(num_nodes, batch_size, distance_threshold, max_edges_per_node=5)"},
    {"build_graph", build_graph, METH_VARARGS, "Construct edge list for each LiDAR scan"},
    {"get_node_positions", get_node_positions, METH_NOARGS, "Get (x,y) node positions"},
    {NULL, NULL, 0, NULL}
};

// Pythonモジュールの定義
static struct PyModuleDef lidargraphmodule = {
    PyModuleDef_HEAD_INIT,
    "lidar_graph",
    "A C module for building graph from LiDAR scan data.",
    -1,
    LidarGraphMethods
};

// Pythonモジュールの初期化関数
PyMODINIT_FUNC PyInit_lidar_graph(void) {
    return PyModule_Create(&lidargraphmodule);
}