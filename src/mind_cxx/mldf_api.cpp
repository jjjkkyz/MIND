#include "mldf_api.h"





int smooth_Fn_with_data(int p1, int p2, int l1, int l2, void* data)
{
    /*
    if (l1 == 0 || l2 == 0) {
        return 100;
    }
    int* myData = (int*)data;

    int a = myData[p1];
    int b = myData[p2];
    if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
        if (l1 != l2) {
            return 0;
        }
        return 10;
    }
    */
    //if ((myData[p1] > 0 && myData[p2] < 0) || (myData[p1] < 0 && myData[p2] > 0)) {
    //    if (l1 == l2) return 10;
    //}
    if (l1 != l2) {
        return 1;
    }
    return 0;
}

int smooth_Fn(int p1, int p2, int l1, int l2, void* data)
{
    float* myData = (float*)data;
    if (l1 == l2) return 0;
    return 1;
}




int dataFn(int p, int l, void* data)
{
    if (l == 0) {
        return 1000;
    }

    std::vector<std::set<int>>& myData = *(reinterpret_cast<std::vector<std::set<int>>*> (data));
    if (myData[p].find(l) != myData[p].end()) {
        return 0;
    }
    if (myData[p].size() == 0) {
        return 0;
    }
    //if (*myData[p].begin() == 25) {
    //    std::cout << std::endl;
    //}
    if (myData[p].size() == 1 && l != *myData[p].begin()) {
        return 1000;
    }
    return 100;
}


std::unordered_map <int, std::set<int>> generate_old_new_map(
    int* labels,
    int* relabels,
    const int64_t sz, const int64_t sy, const int64_t sx
) {
    int cur, bef;
    std::unordered_map <int, std::set<int>> maps;
    const int64_t sxy = sx * sy;
    for (int64_t z = 0; z < sz; z++) {
        for (int64_t y = 0; y < sy; y++) {
            for (int64_t x = 0; x < sx; x++) {
                int64_t loc = x + sx * y + sxy * z;
                cur = relabels[loc];
                bef = labels[loc];
                if (bef == 0) {
                    continue;
                }
                //if (cur == 0) {
                //	continue;
                //}
                if (maps.find(bef) == maps.end()) {
                    maps[bef] = std::set<int>();
                }
                maps[bef].insert(cur);
                //if (cur != 0) {
                //    maps[bef].insert(cur);
                //}
            }
        }
    }
    return maps;
}




void grid_graph_expansion(int* labels, int* relabels, int* sign, int* output, int N, size_t sz, size_t sy, size_t sx) {
    int sxy = sx * sy;
    int neighborhood[26];
    std::unordered_map<int, int> idx_map;
    std::vector<int> inv_idx_map;

    std::unordered_map<int, int> label_map;
    std::vector<int> inv_label_map;
    std::vector<std::set<int>> voxel_label_cost;
    std::vector<int> sign_vec;
    std::vector<int> remark_labels;
    //std::map <int, std::set<int>> old_new = generate_old_new_map(origin_labels, relabels, sz, sy, sx);
    std::unordered_map <int, std::set<int>> old_new;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int fake = labels[loc];
                // any not background label in unorient region is fake, expansion it.
                if (fake != 0) {
                    if (!old_new.contains(fake)) {
                        old_new[fake] = std::set<int>();
                    }
                    if (relabels[loc] != 0) {
                        old_new[fake].insert(relabels[loc]);
                    }
                }
            }
        }
    }
    int nPix = 0;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int fake = labels[loc];
                // ��¼������erode and relabel�б���ʴ����voxel
                if (fake != 0)
                {
                    if (relabels[loc] == 0)
                    {
                        idx_map[loc] = nPix;
                        inv_idx_map.push_back(loc);
                        nPix++;
                        voxel_label_cost.push_back(old_new[fake]);
                        sign_vec.push_back(sign[loc]);
                        remark_labels.push_back(0);
                    }
                    //����Ҳ��Ҫ�뱻��ʴ��voxel�ڽӵ�voxel����grid expansion
                    else {
                        compute_neighborhood_all(neighborhood, x, y, z, sx, sy, sz, 6);
                        for (int i = 0; i < 6; i++) {
                            int neighboridx = loc + neighborhood[i];
                            //���ڱ߽�voxel��neighborhood[i]Ϊ0����unorient_region[loc]�˴���Ϊ0�����Բ�����⿼��
                            if (relabels[neighboridx] != 0) {
                                //�ҵ�����Χ��voxel����ʹ��ǰvoxel��relabel����label������ҲҪ�������뵽grid expansion��
                                idx_map[loc] = nPix;
                                inv_idx_map.push_back(loc);
                                nPix++;
                                voxel_label_cost.push_back(std::set({ relabels[loc] }));
                                remark_labels.push_back(relabels[loc]);
                                sign_vec.push_back(sign[loc]);
                                break;
                            }
                        }
                    }
                }
                
                
                
            }
        }
    }
    
     GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(nPix, N + 1);
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                if (idx_map.contains(loc)) {
                    if (x>0 && idx_map.contains(loc-1)) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - 1]);
                    }
                    if (y>0 && idx_map.contains(loc - sx)) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - sx]);
                    }
                    if (z>0 && idx_map.contains(loc - sxy)) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - sxy]);
                    }
                    gc->setLabel(idx_map[loc], remark_labels[idx_map[loc]]);
                }
            }
        }
    }
    
    gc->setDataCost(&dataFn, (void*)&voxel_label_cost);
    gc->setSmoothCost(&smooth_Fn_with_data, (void*)sign_vec.data());
    gc->expansion(2);
    //std::cout << gc->giveDataEnergy() << std::endl;
    //std::cout << gc->giveSmoothEnergy() << std::endl;
    
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                if (idx_map.contains(loc)) {
                    output[loc] = gc->whatLabel(idx_map[loc]);
                }
                else{
                    output[loc] = relabels[loc];
                }
            }
        }
    }

    //for (int i = 0; i < nPix; i++) {
    //    output[inv_idx_map[i]] = gc->whatLabel(i);
    //}
    delete gc;
}


void grid_graph_cut(int* origin_labels, int* relabels, float* udf, int* output, int N, size_t sz, size_t sy, size_t sx, int loop) {
    int sxy = sx * sy;
    std::unordered_map<int, int> idx_map;
    std::vector<int> inv_idx_map;

    std::unordered_map<int, int> label_map;
    std::vector<int> inv_label_map;
    std::vector<float> udf_list;
    std::unordered_map <int, std::set<int>> old_new = generate_old_new_map(origin_labels, relabels, sz, sy, sx);
    int nPix = 0;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int pre = origin_labels[loc];
                // any not background label
                if (pre != 0) {
                    idx_map[loc] = nPix;
                    inv_idx_map.push_back(loc);
                    nPix++;
                }
            }
        }
    }
    std::vector<std::set<int>> voxel_label_cost;
    int* remark_labels = new int[nPix]();
    int ii = 0;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int pre = origin_labels[loc];
                int now = relabels[loc];
                if (pre != 0) {
                    if (now == 0) {
                        voxel_label_cost.push_back(old_new[pre]);
                    }
                    else {
                        voxel_label_cost.push_back(std::set({ now }));

                    }
                    udf_list.push_back(udf[loc]);
                    // һһ��Ӧ������
                    if (old_new[pre].size() == 1) {
                        remark_labels[ii] = *old_new[pre].begin();
                    }
                    //һ�Զ����һ��0
                    else {
                        remark_labels[ii] = relabels[loc];
                    }

                    ii++;
                }
            }
        }
    }

    std::cout << "begin GC init " << std::endl;
    GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(nPix, N + 1);
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int pre = origin_labels[loc];
                if (pre != 0) {
                    if (x > 0 && origin_labels[loc - 1] != 0) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - 1]);
                    }
                    if (y > 0 && origin_labels[loc - sx] != 0) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - sx]);
                    }
                    if (z > 0 && origin_labels[loc - sxy] != 0) {
                        gc->setNeighbors(idx_map[loc], idx_map[loc - sxy]);
                    }
                    gc->setLabel(idx_map[loc], remark_labels[idx_map[loc]]);
                }
            }
        }
    }
    std::cout << "begin GC cal" << std::endl;
    gc->setDataCost(&dataFn, (void*)&voxel_label_cost);
    gc->setSmoothCost(&smooth_Fn, (void*)udf_list.data());
    gc->expansion(loop);
    //std::cout << gc->giveDataEnergy() << std::endl;
    //std::cout << gc->giveSmoothEnergy() << std::endl;
    std::cout << "end GC " << std::endl;
    for (int i = 0; i < nPix; i++) {
        output[inv_idx_map[i]] = gc->whatLabel(i);
    }
    delete gc;
    delete[] remark_labels;
}



int re_sort(int* labels, size_t sx, size_t sy, size_t sz) {
    int it = 0;
    std::unordered_map<int, int> indexMap;
    int sxy = sx * sy;
    int index = 0;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int cur = labels[loc];
                if (indexMap.find(cur) == indexMap.end()) {
                    indexMap[cur] = index++;
                }
            }
        }
    }

    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int cur = labels[loc];
                labels[loc] = indexMap[cur];
            }
        }
    }
    return indexMap.size();
}


void inplace_label(int* labels, std::unordered_map<int, int> inplace_set, size_t sz, size_t sy, size_t sx, int* output) {
    int neighborhood[26];
    int connectivity = 26;
    int label, record_label;
    int sxy = sx * sy;
    for (int z = 0; z < sz; z++) {
        for (int y = 0; y < sy; y++) {
            for (int x = 0; x < sx; x++) {
                int loc = x + sx * y + sxy * z;
                int cur = labels[loc];
                if (inplace_set.contains(cur)) {
                    output[loc] = inplace_set[cur];
                }
            }
        }
    }
}


std::unordered_map<int, int> label_merge(std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> surface_contact, int re_N,float t) { 
    std::vector<int> vec;
    std::unordered_map<int, int> label_map;
    for (int i = 0; i < re_N + 1; i++) {
        label_map[i] = i;
        vec.push_back(1);
    }

    std::list<std::pair<int, int>> region_graph;
    //std::queue<std::pair<int, int>> region_graph;
    for (auto kv : surface_contact) {
        region_graph.push_back(kv.first);
    }

    while (!region_graph.empty()) {
        auto pick_one = region_graph.begin();
        for (auto it = region_graph.begin(); it != region_graph.end(); ++it) {
            if (vec[it->first] + vec[it->second] < vec[pick_one->first] + vec[pick_one->second]) {
                pick_one = it;
            }
            else if (vec[it->first] + vec[it->second] == vec[pick_one->first] + vec[pick_one->second])
            {
                if (float(surface_contact[*it].second)/float(surface_contact[*it].first + 1) > 
                    float(surface_contact[*pick_one].second) / float(surface_contact[*pick_one].first + 1)) {
                    pick_one = it;
                }
            }
        }
        
        auto id_1 = label_map[pick_one->first];
        auto id_2 = label_map[pick_one->second];
        region_graph.erase(pick_one);

        if (id_1 == id_2) {
            continue;
        }
        // ȷ�����������
        auto kk = std::make_pair(
            std::min(id_1, id_2),
            std::max(id_1, id_2)
        );
        //if (!surface_contact.contains(kk)) {
        //    std::cout << kk.first << " " << kk.second << std::endl;
        //}
        auto vs = surface_contact[kk];
        if (vs.first < 10) {
            std::cout << "No true connect bettwen " << id_1 << " , " << id_2 << std::endl;
            continue;
        }
        if (float(vs.second) / float(vs.first) > t) {
            std::cout << "Cross point connect too large bettwen " << id_1 << " , " << id_2 << ". " << float(vs.second) / float(vs.first) << std::endl;
            continue;
        }
        std::cout << "Merging!!!!!!!!!!!!! " << id_1 << " , " << id_2 << ". "  << float(vs.second) << " " << float(vs.first) << " " << float(vs.second) / float(vs.first) << std::endl;
        int weight = vec[label_map[id_1]];
        for (int i = 0; i < re_N + 1; i++) {
            if (label_map[i] == id_2) {
                label_map[i] = id_1;
                vec[i] = weight + 1;
            }
            if (label_map[i] == id_1) {
                vec[i] = weight + 1;
            }
        }


        surface_contact.erase(kk);

        std::vector<std::pair<int, int>> current_k;
        for (auto kv : surface_contact) {
            current_k.push_back(kv.first);
        }
        for (auto k : current_k) {
            if (k.first == id_2) {
                auto another = k.second;
                auto v = surface_contact[k];
                surface_contact.erase(k);
                auto new_k = std::make_pair(
                    std::min(id_1, another),
                    std::max(id_1, another)
                );
                if (surface_contact.contains(new_k)) {
                    auto old_v = surface_contact[new_k];
                    surface_contact[new_k] = std::pair<int, int>(old_v.first + v.first, old_v.second + v.second);
                }
                else
                {
                    surface_contact[new_k] = v;
                    region_graph.push_back(new_k);
                }


            }

            if (k.second == id_2) {


                auto another = k.first;
                auto v = surface_contact[k];
                surface_contact.erase(k);
                auto new_k = std::make_pair(
                    std::min(id_1, another),
                    std::max(id_1, another)
                );
                if (surface_contact.contains(new_k)) {
                    auto old_v = surface_contact[new_k];
                    surface_contact[new_k] = std::pair<int, int>(old_v.first + v.first, old_v.second + v.second);
                }
                else
                {
                    surface_contact[new_k] = v;
                    region_graph.push_back(new_k);
                }
            }
        }
    }
    return label_map;
}
