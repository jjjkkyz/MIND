#pragma once

#include <unordered_map>
#include <set>


//struct pair_hash {
//	inline std::size_t operator()(const std::pair<uint64_t, uint64_t>& v) const {
//		return v.first * 31 + v.second; // arbitrary hash fn
//	}
//};

struct pair_hash {
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1, T2>& p) const {
		// 正确实现哈希函数（如结合两个值的哈希）
		auto hash1 = std::hash<T1>{}(p.first);
		auto hash2 = std::hash<T2>{}(p.second);
		return hash1 ^ (hash2 << 1); // 或其他组合方式
	}
};


void compute_neighborhood(
	int* neighborhood,
	const int x, const int y, const int z,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const int connectivity
);

void compute_neighborhood_all(
	int* neighborhood,
	const int x, const int y, const int z,
	const int sx, const int sy, const int sz,
	const int connectivity = 26
);

// 自定义比较函数
struct pair_equal {
	bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
		// 仅比较第一个元素（自定义逻辑）
		return (a.first == b.first)&&(a.second == b.second);
	}
};


const std::unordered_map<std::pair<int, int>, float, pair_hash, pair_equal>
extract_region_graph(
	int* labels,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const float wx = 1, const float wy = 1, const float wz = 1,
	const int64_t connectivity = 26,
	const bool surface_area = true
);



int* _erode_ml(
	int* labels,
	int* mask,
	const int64_t sx, const int64_t sy, const int64_t sz,
	const int64_t connectivity
);



int* erode_ml_c(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity, const int64_t it);


int* _dilate_ml(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity = 26
);


int* dilate_ml_c(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity = 26, const int64_t it = 1);


const std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal>
extract_region_graph_background(
	int* labels,
	int* inner_mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity = 6
);

const std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> 
extract_region_graph_cross(
	int* labels,
	int* sign,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity
);

bool detect_bound(int* input, size_t sz, size_t sy, size_t sx);