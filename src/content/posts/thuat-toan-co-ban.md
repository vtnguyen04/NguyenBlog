---
title: "Các thuật toán cơ bản mà mọi lập trình viên cần biết"
published: 2025-07-07
description: "Tổng hợp các thuật toán cơ bản quan trọng trong lập trình và phỏng vấn"
tags: [Thuật toán, c++, Data Structures, Interview]
category: "Thuật toán"
draft: false
lang: "vi"
---

## Mục lục

1. [Thuật toán tìm kiếm](#1-thuật-toán-tìm-kiếm)
    - [Linear Search (Tìm kiếm tuyến tính)](#linear-search-tìm-kiếm-tuyến-tính)
    - [Binary Search (Tìm kiếm nhị phân)](#binary-search-tìm-kiếm-nhị-phân)
2. [Thuật toán sắp xếp](#2-thuật-toán-sắp-xếp)
    - [Bubble Sort](#bubble-sort)
    - [Quick Sort](#quick-sort)
    - [Merge Sort](#merge-sort)
3. [Cấu trúc dữ liệu cơ bản](#3-cấu-trúc-dữ-liệu-cơ-bản)
    - [Stack (Ngăn xếp)](#stack-ngăn-xếp)
    - [Queue (Hàng đợi)](#queue-hàng-đợi)
4. [Thuật toán đồ thị](#4-thuật-toán-đồ-thị)
    - [Depth-First Search (DFS)](#depth-first-search-dfs)
    - [Breadth-First Search (BFS)](#breadth-first-search-bfs)
5. [Thuật toán quy hoạch động (Dynamic Programming)](#5-thuật-toán-quy-hoạch-động-dynamic-programming)
    - [Fibonacci với memoization](#fibonacci-với-memoization)
    - [Longest Common Subsequence (LCS)](#longest-common-subsequence-lcs)
6. [Thuật toán tham lam (Greedy)](#6-thuật-toán-tham-lam-greedy)
    - [Activity Selection Problem](#activity-selection-problem)
7. [Thuật toán chia để trị (Divide and Conquer)](#7-thuật-toán-chia-để-trị-divide-and-conquer)
    - [Strassen's Matrix Multiplication](#strassens-matrix-multiplication)
8. [Thuật toán đồ thị & cấu trúc dữ liệu nâng cao](#8-thuật-toán-đồ-thị--cấu-trúc-dữ-liệu-nâng-cao)
    - [Dijkstra - Đường đi ngắn nhất trên đồ thị có trọng số dương](#dijkstra---đường-đi-ngắn-nhất-trên-đồ-thị-có-trọng-số-dương)
    - [Bellman-Ford - Đường đi ngắn nhất với trọng số âm](#bellman-ford---đường-đi-ngắn-nhất-với-trọng-số-âm)
    - [Floyd-Warshall - Đường đi ngắn nhất giữa mọi cặp đỉnh](#floyd-warshall---đường-đi-ngắn-nhất-giữa-mọi-cặp-đỉnh)
    - [Kruskal - Cây khung nhỏ nhất (Minimum Spanning Tree)](#kruskal---cây-khung-nhỏ-nhất-minimum-spanning-tree)
    - [Edmonds-Karp - Luồng cực đại (Maximum Flow)](#edmonds-karp---luồng-cực-đại-maximum-flow)
    - [Heap (Min Heap)](#heap-min-heap)
    - [Union-Find (Disjoint Set)](#union-find-disjoint-set)
    - [Trie (Cây tiền tố)](#trie-cây-tiền-tố)
    - [Topological Sort (Sắp xếp topo)](#topological-sort-sắp-xếp-topo)
    - [KMP - Tìm kiếm chuỗi](#kmp---tìm-kiếm-chuỗi)
    - [Aho-Corasick - Tìm kiếm nhiều mẫu](#aho-corasick---tìm-kiếm-nhiều-mẫu)
    - [Splay Tree](#splay-tree)
    - [Segment Tree (Cây đoạn)](#segment-tree-cây-đoạn)
9. [Binary Tree (Cây nhị phân)](#binary-tree-cây-nhị-phân)
10. [Linked List (Danh sách liên kết đơn)](#linked-list-danh-sách-liên-kết-đơn)
11. [Kết luận](#kết-luận)


# Các thuật toán cơ bản mà mọi lập trình viên cần biết 🧮

Thuật toán là nền tảng của lập trình và khoa học máy tính. Trong bài viết này, tôi sẽ chia sẻ những thuật toán cơ bản quan trọng mà mọi lập trình viên nên nắm vững.

## 1. Thuật toán tìm kiếm

### Linear Search (Tìm kiếm tuyến tính)

```cpp
#include <vector>
using namespace std;

// Tìm kiếm tuyến tính trong mảng
int linear_search(const vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) return i;
    }
    return -1;
}

// Độ phức tạp: O(n)
// Không gian: O(1)
```

### Binary Search (Tìm kiếm nhị phân)

```cpp
#include <vector>
using namespace std;

// Tìm kiếm nhị phân trong mảng đã sắp xếp
int binary_search(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Độ phức tạp: O(log n)
// Không gian: O(1)
// Yêu cầu: Mảng đã được sắp xếp
```

## 2. Thuật toán sắp xếp

### Bubble Sort

```cpp
#include <vector>
using namespace std;

// Sắp xếp nổi bọt
void bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
// Độ phức tạp: O(n^2)
// Không gian: O(1)
```

### Quick Sort

```cpp
#include <vector>
using namespace std;

// Sắp xếp nhanh (Quick Sort)
void quick_sort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int pivot = arr[left + (right - left) / 2];
    int i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) ++i;
        while (arr[j] > pivot) --j;
        if (i <= j) {
            swap(arr[i], arr[j]);
            ++i; --j;
        }
    }
    if (left < j) quick_sort(arr, left, j);
    if (i < right) quick_sort(arr, i, right);
}
// Độ phức tạp trung bình: O(n log n)
// Độ phức tạp xấu nhất: O(n^2)
// Không gian: O(log n)
```

### Merge Sort

```cpp
#include <vector>
using namespace std;

// Hợp hai mảng con đã sắp xếp
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; ++i) L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j) R[j] = arr[mid + 1 + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// Sắp xếp trộn (Merge Sort)
void merge_sort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}
// Độ phức tạp: O(n log n)
// Không gian: O(n)
```

## 3. Cấu trúc dữ liệu cơ bản

### Stack (Ngăn xếp)

```cpp
#include <vector>
#include <iostream>
using namespace std;

// Cấu trúc dữ liệu ngăn xếp (Stack)
class Stack {
    vector<int> items;
public:
    void push(int item) { items.push_back(item); }
    void pop() { if (!is_empty()) items.pop_back(); }
    int top() const { return is_empty() ? -1 : items.back(); }
    bool is_empty() const { return items.empty(); }
    int size() const { return items.size(); }
};

// Độ phức tạp các thao tác: O(1)
```

### Queue (Hàng đợi)

```cpp
#include <queue>
using namespace std;

// Cấu trúc dữ liệu hàng đợi (Queue)
class Queue {
    queue<int> items;
public:
    void enqueue(int item) { items.push(item); }
    void dequeue() { if (!is_empty()) items.pop(); }
    int front() const { return is_empty() ? -1 : items.front(); }
    bool is_empty() const { return items.empty(); }
    int size() const { return items.size(); }
};

// Độ phức tạp các thao tác: O(1)
```

## 4. Thuật toán đồ thị

### Depth-First Search (DFS)

```cpp
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Graph {
private:
    std::unordered_map<int, std::vector<int>> adjacencyList;
    
public:
    void addEdge(int from, int to) {
        adjacencyList[from].push_back(to);
        adjacencyList[to].push_back(from); // Undirected graph
    }
    
    void dfs(int start) {
        std::unordered_set<int> visited;
        dfsHelper(start, visited);
    }
    
private:
    void dfsHelper(int node, std::unordered_set<int>& visited) {
        visited.insert(node);
        std::cout << node << " ";
        
        for (int neighbor : adjacencyList[node]) {
            if (visited.find(neighbor) == visited.end()) {
                dfsHelper(neighbor, visited);
            }
        }
    }
};

// Độ phức tạp: O(V + E)
// V: số đỉnh, E: số cạnh
```

### Breadth-First Search (BFS)

```cpp
#include <queue>

void Graph::bfs(int start) {
    std::unordered_set<int> visited;
    std::queue<int> q;
    
    visited.insert(start);
    q.push(start);
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        std::cout << current << " ";
        
        for (int neighbor : adjacencyList[current]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
}

// Độ phức tạp: O(V + E)
// V: số đỉnh, E: số cạnh
```

## 5. Thuật toán quy hoạch động (Dynamic Programming)

### Fibonacci với memoization

```cpp
#include <vector>
using namespace std;

// Tính Fibonacci sử dụng memoization (quy hoạch động)
int fibonacci_memo(int n, vector<int>& memo) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];
    return memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo);
}
// Sử dụng: vector<int> memo(n+1, -1); fibonacci_memo(n, memo);
// Độ phức tạp: O(n)
// Không gian: O(n)
```

### Longest Common Subsequence (LCS)

```cpp
#include <vector>
#include <string>
using namespace std;

// Tìm độ dài dãy con chung dài nhất (LCS)
int lcs(const string& s1, const string& s2) {
    int m = s1.size(), n = s2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (s1[i-1] == s2[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m][n];
}
// Độ phức tạp: O(m*n)
// Không gian: O(m*n)
```

## 6. Thuật toán tham lam (Greedy)

### Activity Selection Problem

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// Bài toán chọn hoạt động (Activity Selection)
struct Activity {
    int start, finish;
};

bool cmp(const Activity& a, const Activity& b) {
    return a.finish < b.finish;
}

vector<int> activity_selection(vector<Activity>& activities) {
    int n = activities.size();
    sort(activities.begin(), activities.end(), cmp);
    vector<int> selected;
    selected.push_back(0); // chọn hoạt động đầu tiên
    int last = 0;
    for (int i = 1; i < n; ++i) {
        if (activities[i].start >= activities[last].finish) {
            selected.push_back(i);
            last = i;
        }
    }
    return selected;
}
// Độ phức tạp: O(n log n) (do sắp xếp)
// Không gian: O(1)
```

## 7. Thuật toán chia để trị (Divide and Conquer)

### Strassen's Matrix Multiplication

```cpp
#include <vector>
using namespace std;

typedef vector<vector<int>> Matrix;

// Cộng hai ma trận vuông
Matrix matrix_add(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// Trừ hai ma trận vuông
Matrix matrix_subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Nhân ma trận bằng thuật toán Strassen
Matrix strassen_multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n == 1) return Matrix{{A[0][0] * B[0][0]}};
    int mid = n / 2;
    // Chia ma trận thành 4 phần
    Matrix A11(mid, vector<int>(mid)), A12(mid, vector<int>(mid)),
           A21(mid, vector<int>(mid)), A22(mid, vector<int>(mid)),
           B11(mid, vector<int>(mid)), B12(mid, vector<int>(mid)),
           B21(mid, vector<int>(mid)), B22(mid, vector<int>(mid));
    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + mid];
            A21[i][j] = A[i + mid][j];
            A22[i][j] = A[i + mid][j + mid];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + mid];
            B21[i][j] = B[i + mid][j];
            B22[i][j] = B[i + mid][j + mid];
        }
    }
    // 7 phép nhân con
    Matrix P1 = strassen_multiply(A11, matrix_subtract(B12, B22));
    Matrix P2 = strassen_multiply(matrix_add(A11, A12), B22);
    Matrix P3 = strassen_multiply(matrix_add(A21, A22), B11);
    Matrix P4 = strassen_multiply(A22, matrix_subtract(B21, B11));
    Matrix P5 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22));
    Matrix P6 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22));
    Matrix P7 = strassen_multiply(matrix_subtract(A11, A21), matrix_add(B11, B12));
    // Tính các phần tử của ma trận kết quả
    Matrix C11 = matrix_add(matrix_subtract(matrix_add(P5, P4), P2), P6);
    Matrix C12 = matrix_add(P1, P2);
    Matrix C21 = matrix_add(P3, P4);
    Matrix C22 = matrix_subtract(matrix_subtract(matrix_add(P5, P1), P3), P7);
    // Ghép các phần lại
    Matrix C(n, vector<int>(n));
    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + mid] = C12[i][j];
            C[i + mid][j] = C21[i][j];
            C[i + mid][j + mid] = C22[i][j];
        }
    }
    return C;
}
// Độ phức tạp: O(n^2.807)
```

## 8. Thuật toán đồ thị & cấu trúc dữ liệu nâng cao

### Dijkstra - Đường đi ngắn nhất trên đồ thị có trọng số dương

```cpp
#include <vector>
#include <queue>
#include <climits>

struct Edge {
    int to;
    int weight;
    
    Edge(int t, int w) : to(t), weight(w) {}
};

class WeightedGraph {
private:
    std::vector<std::vector<Edge>> graph;
    int vertices;
    
public:
    WeightedGraph(int v) : vertices(v) {
        graph.resize(v);
    }
    
    void addEdge(int from, int to, int weight) {
        graph[from].push_back(Edge(to, weight));
        graph[to].push_back(Edge(from, weight)); // Undirected
    }
    
    std::vector<int> dijkstra(int start) {
        std::vector<int> dist(vertices, INT_MAX);
        std::priority_queue<std::pair<int, int>, 
                          std::vector<std::pair<int, int>>, 
                          std::greater<std::pair<int, int>>> pq;
        
        dist[start] = 0;
        pq.push({0, start});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            for (const Edge& edge : graph[u]) {
                int v = edge.to;
                int weight = edge.weight;
                
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        
        return dist;
    }
};
// Độ phức tạp: O((V+E) log V)
```

### Bellman-Ford - Đường đi ngắn nhất với trọng số âm

```cpp
#include <vector>
#include <limits>
using namespace std;

struct EdgeBF {
    int u, v, w;
};

vector<int> bellman_ford(int n, const vector<EdgeBF>& edges, int start) {
    vector<int> dist(n, numeric_limits<int>::max());
    dist[start] = 0;
    for (int i = 1; i < n; ++i) {
        for (const auto& e : edges) {
            if (dist[e.u] < numeric_limits<int>::max() && dist[e.v] > dist[e.u] + e.w)
                dist[e.v] = dist[e.u] + e.w;
        }
    }
    // Kiểm tra chu trình âm
    for (const auto& e : edges) {
        if (dist[e.u] < numeric_limits<int>::max() && dist[e.v] > dist[e.u] + e.w)
            throw runtime_error("Đồ thị có chu trình âm");
    }
    return dist;
}
// Độ phức tạp: O(V*E)
```

### Floyd-Warshall - Đường đi ngắn nhất giữa mọi cặp đỉnh

```cpp
#include <vector>
#include <climits>

std::vector<std::vector<int>> floydWarshall(const std::vector<std::vector<int>>& graph) {
    int n = graph.size();
    std::vector<std::vector<int>> dist = graph;
    
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    
    return dist;
}
// Độ phức tạp: O(V^3)
```



### Kruskal - Cây khung nhỏ nhất (Minimum Spanning Tree)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v, w;
    bool operator<(const Edge& other) const { return w < other.w; }
};

struct DSU {
    vector<int> parent, rank;
    DSU(int n) : parent(n), rank(n, 0) { for (int i = 0; i < n; ++i) parent[i] = i; }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) ++rank[x];
        return true;
    }
};

int kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());
    DSU dsu(n);
    int mst_weight = 0;
    for (const Edge& e : edges) {
        if (dsu.unite(e.u, e.v)) mst_weight += e.w;
    }
    return mst_weight;
}
// Độ phức tạp: O(E log E)
```

### Edmonds-Karp - Luồng cực đại (Maximum Flow)

```cpp
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

int edmonds_karp(int n, vector<vector<int>>& capacity, int s, int t) {
    vector<vector<int>> flow(n, vector<int>(n, 0));
    int max_flow = 0;
    while (true) {
        vector<int> parent(n, -1);
        queue<int> q;
        q.push(s);
        parent[s] = s;
        while (!q.empty() && parent[t] == -1) {
            int u = q.front(); q.pop();
            for (int v = 0; v < n; ++v) {
                if (parent[v] == -1 && capacity[u][v] - flow[u][v] > 0) {
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        if (parent[t] == -1) break;
        int aug = 1e9;
        for (int v = t; v != s; v = parent[v])
            aug = min(aug, capacity[parent[v]][v] - flow[parent[v]][v]);
        for (int v = t; v != s; v = parent[v]) {
            flow[parent[v]][v] += aug;
            flow[v][parent[v]] -= aug;
        }
        max_flow += aug;
    }
    return max_flow;
}
// Độ phức tạp: O(V*E^2)
```


### Heap (Min Heap)

```cpp
#include <queue>
#include <vector>
using namespace std;

// Min Heap sử dụng priority_queue
priority_queue<int, vector<int>, greater<int>> min_heap;
// Các thao tác: push(x), pop(), top(), empty()
// Độ phức tạp: O(log n) cho push/pop
```

### Union-Find (Disjoint Set)

```cpp
#include <vector>
using namespace std;

struct UnionFind {
    vector<int> parent, rank;
    UnionFind(int n) : parent(n), rank(n, 0) { for (int i = 0; i < n; ++i) parent[i] = i; }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    void unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return;
        if (rank[x] < rank[y]) swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) ++rank[x];
    }
    bool same(int x, int y) { return find(x) == find(y); }
};
// Độ phức tạp: gần O(1) cho mỗi thao tác
```

### Trie (Cây tiền tố)

```cpp
#include <string>
#include <vector>
using namespace std;

struct TrieNode {
    vector<TrieNode*> children;
    bool is_end;
    TrieNode() : children(26, nullptr), is_end(false) {}
};

struct Trie {
    TrieNode* root;
    Trie() : root(new TrieNode()) {}
    void insert(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) node->children[idx] = new TrieNode();
            node = node->children[idx];
        }
        node->is_end = true;
    }
    bool search(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return node->is_end;
    }
};
// Độ phức tạp: O(L) với L là độ dài từ
```

### Topological Sort (Sắp xếp topo)

```cpp
#include <vector>
#include <queue>
using namespace std;

vector<int> topo_sort(int n, const vector<vector<int>>& adj) {
    vector<int> indeg(n, 0);
    for (int u = 0; u < n; ++u)
        for (int v : adj[u]) ++indeg[v];
    queue<int> q;
    for (int i = 0; i < n; ++i) if (indeg[i] == 0) q.push(i);
    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (--indeg[v] == 0) q.push(v);
        }
    }
    return order;
}
// Độ phức tạp: O(V + E)
```

### KMP - Tìm kiếm chuỗi

```cpp
#include <vector>
#include <string>
using namespace std;

vector<int> kmp_prefix(const string& pattern) {
    int n = pattern.size();
    vector<int> pi(n, 0);
    for (int i = 1, j = 0; i < n; ++i) {
        while (j > 0 && pattern[i] != pattern[j]) j = pi[j-1];
        if (pattern[i] == pattern[j]) ++j;
        pi[i] = j;
    }
    return pi;
}

vector<int> kmp_search(const string& text, const string& pattern) {
    vector<int> pi = kmp_prefix(pattern), res;
    int n = text.size(), m = pattern.size();
    for (int i = 0, j = 0; i < n; ++i) {
        while (j > 0 && text[i] != pattern[j]) j = pi[j-1];
        if (text[i] == pattern[j]) ++j;
        if (j == m) {
            res.push_back(i - m + 1);
            j = pi[j-1];
        }
    }
    return res;
}
// Độ phức tạp: O(n + m)
```


### Aho-Corasick - Tìm kiếm nhiều mẫu

```cpp
#include <vector>
#include <queue>
#include <string>
using namespace std;

struct ACNode {
    vector<ACNode*> children;
    ACNode* fail;
    vector<int> output;
    ACNode() : children(26, nullptr), fail(nullptr) {}
};

struct AhoCorasick {
    ACNode* root;
    AhoCorasick() : root(new ACNode()) {}
    void insert(const string& word, int id) {
        ACNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) node->children[idx] = new ACNode();
            node = node->children[idx];
        }
        node->output.push_back(id);
    }
    void build() {
        queue<ACNode*> q;
        root->fail = root;
        for (int i = 0; i < 26; ++i) {
            if (root->children[i]) {
                root->children[i]->fail = root;
                q.push(root->children[i]);
            } else {
                root->children[i] = root;
            }
        }
        while (!q.empty()) {
            ACNode* node = q.front(); q.pop();
            for (int i = 0; i < 26; ++i) {
                if (node->children[i]) {
                    ACNode* f = node->fail;
                    while (!f->children[i]) f = f->fail;
                    node->children[i]->fail = f->children[i];
                    node->children[i]->output.insert(
                        node->children[i]->output.end(),
                        f->children[i]->output.begin(),
                        f->children[i]->output.end()
                    );
                    q.push(node->children[i]);
                }
            }
        }
    }
    // Hàm tìm kiếm trả về vị trí xuất hiện của các mẫu
    vector<pair<int, int>> search(const string& text) {
        vector<pair<int, int>> res;
        ACNode* node = root;
        for (int i = 0; i < text.size(); ++i) {
            int idx = text[i] - 'a';
            while (!node->children[idx]) node = node->fail;
            node = node->children[idx];
            for (int id : node->output) res.emplace_back(i, id);
        }
        return res;
    }
};
// Độ phức tạp: O(N + M + Z) với N: tổng độ dài mẫu, M: độ dài văn bản, Z: số lần khớp
```

### Splay Tree

```cpp
#include <iostream>
using namespace std;

struct SplayNode {
    int key;
    SplayNode *left, *right, *parent;
    SplayNode(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
};

void rotate(SplayNode*& root, SplayNode* x) {
    SplayNode* p = x->parent;
    if (!p) return;
    SplayNode* g = p->parent;
    if (p->left == x) {
        p->left = x->right;
        if (x->right) x->right->parent = p;
        x->right = p;
    } else {
        p->right = x->left;
        if (x->left) x->left->parent = p;
        x->left = p;
    }
    p->parent = x;
    x->parent = g;
    if (g) {
        if (g->left == p) g->left = x;
        else g->right = x;
    } else {
        root = x;
    }
}

void splay(SplayNode*& root, SplayNode* x) {
    while (x->parent) {
        SplayNode* p = x->parent;
        SplayNode* g = p->parent;
        if (!g) {
            rotate(root, x);
        } else if ((g->left == p) == (p->left == x)) {
            rotate(root, p);
            rotate(root, x);
        } else {
            rotate(root, x);
            rotate(root, x);
        }
    }
}
// Độ phức tạp trung bình: O(log n) cho mỗi thao tác
```

### Segment Tree (Cây đoạn)

```cpp
#include <vector>
using namespace std;

struct SegmentTree {
    int n;
    vector<int> tree;
    SegmentTree(const vector<int>& a) {
        n = a.size();
        tree.resize(4 * n);
        build(a, 1, 0, n - 1);
    }
    void build(const vector<int>& a, int v, int l, int r) {
        if (l == r) tree[v] = a[l];
        else {
            int m = (l + r) / 2;
            build(a, v * 2, l, m);
            build(a, v * 2 + 1, m + 1, r);
            tree[v] = tree[v * 2] + tree[v * 2 + 1];
        }
    }
    int query(int v, int l, int r, int ql, int qr) {
        if (ql > r || qr < l) return 0;
        if (ql <= l && r <= qr) return tree[v];
        int m = (l + r) / 2;
        return query(v * 2, l, m, ql, qr) + query(v * 2 + 1, m + 1, r, ql, qr);
    }
    void update(int v, int l, int r, int pos, int val) {
        if (l == r) tree[v] = val;
        else {
            int m = (l + r) / 2;
            if (pos <= m) update(v * 2, l, m, pos, val);
            else update(v * 2 + 1, m + 1, r, pos, val);
            tree[v] = tree[v * 2] + tree[v * 2 + 1];
        }
    }
};
// Độ phức tạp: O(log n) cho mỗi truy vấn/cập nhật
```

### Binary Tree (Cây nhị phân)

```cpp
template<typename T>
struct TreeNode {
    T data;
    TreeNode* left;
    TreeNode* right;
    
    TreeNode(const T& val) : data(val), left(nullptr), right(nullptr) {}
};

template<typename T>
class BinaryTree {
private:
    TreeNode<T>* root;
    
    void destroyTree(TreeNode<T>* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
    
    void inorderTraversal(TreeNode<T>* node) const {
        if (node) {
            inorderTraversal(node->left);
            std::cout << node->data << " ";
            inorderTraversal(node->right);
        }
    }
    
    void preorderTraversal(TreeNode<T>* node) const {
        if (node) {
            std::cout << node->data << " ";
            preorderTraversal(node->left);
            preorderTraversal(node->right);
        }
    }
    
    void postorderTraversal(TreeNode<T>* node) const {
        if (node) {
            postorderTraversal(node->left);
            postorderTraversal(node->right);
            std::cout << node->data << " ";
        }
    }
    
public:
    BinaryTree() : root(nullptr) {}
    
    ~BinaryTree() {
        destroyTree(root);
    }
    
    void insert(const T& val) {
        root = insertNode(root, val);
    }
    
    TreeNode<T>* insertNode(TreeNode<T>* node, const T& val) {
        if (!node) {
            return new TreeNode<T>(val);
        }
        
        if (val < node->data) {
            node->left = insertNode(node->left, val);
        } else if (val > node->data) {
            node->right = insertNode(node->right, val);
        }
        
        return node;
    }
    
    void inorder() const {
        inorderTraversal(root);
        std::cout << std::endl;
    }
    
    void preorder() const {
        preorderTraversal(root);
        std::cout << std::endl;
    }
    
    void postorder() const {
        postorderTraversal(root);
        std::cout << std::endl;
    }
};
// Độ phức tạp: O(nlogn) với n là số nút
```

### Linked List (Danh sách liên kết đơn)

```cpp
template<typename T>
struct ListNode {
    T data;
    ListNode* next;
    
    ListNode(const T& val) : data(val), next(nullptr) {}
};

template<typename T>
class LinkedList {
private:
    ListNode<T>* head;
    
public:
    LinkedList() : head(nullptr) {}
    
    ~LinkedList() {
        clear();
    }
    
    void insert(const T& val) {
        ListNode<T>* newNode = new ListNode<T>(val);
        newNode->next = head;
        head = newNode;
    }
    
    void remove(const T& val) {
        if (!head) return;
        
        if (head->data == val) {
            ListNode<T>* temp = head;
            head = head->next;
            delete temp;
            return;
        }
        
        ListNode<T>* current = head;
        while (current->next && current->next->data != val) {
            current = current->next;
        }
        
        if (current->next) {
            ListNode<T>* temp = current->next;
            current->next = current->next->next;
            delete temp;
        }
    }
    
    bool search(const T& val) const {
        ListNode<T>* current = head;
        while (current) {
            if (current->data == val) {
                return true;
            }
            current = current->next;
        }
        return false;
    }
    
    void clear() {
        while (head) {
            ListNode<T>* temp = head;
            head = head->next;
            delete temp;
        }
    }
};
// Độ phức tạp: O(n) với n là số node
```

## Kết luận

Những thuật toán cơ bản này là nền tảng quan trọng trong lập trình và khoa học máy tính. Việc hiểu rõ và thực hành các thuật toán này sẽ giúp bạn:

1. **Tối ưu hóa code**: Chọn thuật toán phù hợp cho từng bài toán
2. **Phỏng vấn**: Chuẩn bị tốt cho các cuộc phỏng vấn kỹ thuật
3. **Tư duy logic**: Phát triển khả năng giải quyết vấn đề
4. **Nghiên cứu AI**: Hiểu sâu các thuật toán machine learning

### Lời khuyên:

- **Thực hành thường xuyên**: Code lại các thuật toán nhiều lần
- **Phân tích độ phức tạp**: Hiểu rõ Big O notation
- **So sánh hiệu suất**: Test với các bộ dữ liệu khác nhau
- **Áp dụng thực tế**: Tìm hiểu cách sử dụng trong các framework

Hãy tiếp tục học tập và khám phá thêm các thuật toán nâng cao! 