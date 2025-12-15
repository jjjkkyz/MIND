#include <map>
#include <set>
#include <unordered_map>
#include <iostream>
#include <queue>
#include <list>
#include "gco-v3.0/GCoptimization.h"
#include "m3c.h"
#include "utils.h"




void grid_graph_cut(int* origin_labels, int* relabels, float* udf, int* output, int N, size_t sz, size_t sy, size_t sx, int loop=2);

void grid_graph_expansion(int* orient_region, int* unorient_region, int* sign, int* output, int N, size_t sz, size_t sy, size_t sx);

int re_sort(int* labels, size_t sx, size_t sy, size_t sz);


void inplace_label(int* labels, std::unordered_map<int, int> inplace_set, size_t sz, size_t sy, size_t sx, int* output);


//std::map <int, std::set<int>> generate_old_new_map(
//    int* labels,
//    int* relabels,
//    const int64_t sz, const int64_t sy, const int64_t sx
//);


std::unordered_map<int, int> label_merge(std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> surface_contact, int re_N, float t=1.0f);