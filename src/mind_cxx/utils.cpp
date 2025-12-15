#include"utils.h"
#include <iostream>




void compute_neighborhood(
	int* neighborhood,
	const int x, const int y, const int z,
	const size_t sx, const size_t sy, const size_t sz,
	const int connectivity
) {

	const int sxy = sx * sy;

	const int plus_x = (x < (static_cast<int>(sx) - 1)); // +x
	const int minus_x = -1 * (x > 0); // -x
	const int plus_y = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
	const int minus_y = -static_cast<int>(sx) * (y > 0); // -y
	const int minus_z = -sxy * static_cast<int>(z > 0); // -z

	// 6-hood
	neighborhood[0] = minus_x;
	neighborhood[1] = minus_y;
	neighborhood[2] = minus_z;

	// 18-hood

	// xy diagonals
	neighborhood[3] = (connectivity > 6) * (minus_x + minus_y) * (minus_x && minus_y); // up-left
	neighborhood[4] = (connectivity > 6) * (plus_x + minus_y) * (plus_x && minus_y); // up-right

	// yz diagonals
	neighborhood[5] = (connectivity > 6) * (minus_x + minus_z) * (minus_x && minus_z); // down-left
	neighborhood[6] = (connectivity > 6) * (plus_x + minus_z) * (plus_x && minus_z); // down-right

	// xz diagonals
	neighborhood[7] = (connectivity > 6) * (minus_y + minus_z) * (minus_y && minus_z); // down-left
	neighborhood[8] = (connectivity > 6) * (plus_y + minus_z) * (plus_y && minus_z); // down-right

	// 26-hood

	// Now the eight corners of the cube
	neighborhood[9] = (connectivity > 18) * (minus_x + minus_y + minus_z) * (minus_y && minus_z);
	neighborhood[10] = (connectivity > 18) * (plus_x + minus_y + minus_z) * (minus_y && minus_z);
	neighborhood[11] = (connectivity > 18) * (minus_x + plus_y + minus_z) * (plus_y && minus_z);
	neighborhood[12] = (connectivity > 18) * (plus_x + plus_y + minus_z) * (plus_y && minus_z);
}

void compute_neighborhood_all(
	int* neighborhood,
	const int x, const int y, const int z,
	const int sx, const int sy, const int sz,
	const int connectivity
) {

	const int sxy = sx * sy;

	// 6-hood
	neighborhood[0] = -1 * (x > 0); // -x
	neighborhood[1] = -sx * (y > 0); // -y
	neighborhood[2] = -sxy * (z > 0); // -z

	// 6-hood
	neighborhood[3] = 1 * (x < sx - 1); // +x
	neighborhood[4] = sx * (y < sy - 1); // +y
	neighborhood[5] = sxy * (z < sz - 1); // +z

	// 18-hood

	// xy diagonals
	neighborhood[6] = (connectivity > 6) * (x > 0 && y > 0) * (-1 - sx); // up-left
	neighborhood[7] = (connectivity > 6) * (x < sx - 1 && y > 0) * (1 - sx); // up-right

	// yz diagonals
	neighborhood[8] = (connectivity > 6) * (y > 0 && z > 0) * (-sx - sxy); // down-left
	neighborhood[9] = (connectivity > 6) * (y < sy - 1 && z > 0) * (sx - sxy); // down-right

	// xz diagonals
	neighborhood[10] = (connectivity > 6) * (x > 0 && z > 0) * (-1 - sxy); // down-left
	neighborhood[11] = (connectivity > 6) * (x < sx - 1 && z > 0) * (1 - sxy); // down-right

	// 18-hood

	// xy diagonals
	neighborhood[12] = (connectivity > 6) * (x > 0 && y < sy - 1) * (-1 + sx); // up-left
	neighborhood[13] = (connectivity > 6) * (x < sx - 1 && y < sy - 1) * (1 + sx); // up-right

	// yz diagonals
	neighborhood[14] = (connectivity > 6) * (y > 0 && z < sz - 1) * (-sx + sxy); // down-left
	neighborhood[15] = (connectivity > 6) * (y < sy - 1 && z < sz - 1) * (sx + sxy); // down-right

	// xz diagonals
	neighborhood[16] = (connectivity > 6) * (x > 0 && z < sz - 1) * (-1 + sxy); // down-left
	neighborhood[17] = (connectivity > 6) * (x < sx - 1 && z < sz - 1) * (1 + sxy); // down-right

	// 26-hood

	// Now the four corners of the bottom plane
	neighborhood[18] = (connectivity > 18) * (x > 0 && y > 0 && z > 0) * (-1 - sx - sxy);
	neighborhood[19] = (connectivity > 18) * (x < sx - 1 && y > 0 && z > 0) * (1 - sx - sxy);
	neighborhood[20] = (connectivity > 18) * (x > 0 && y < sy - 1 && z > 0) * (-1 + sx - sxy);
	neighborhood[21] = (connectivity > 18) * (x < sx - 1 && y < sy - 1 && z > 0) * (1 + sx - sxy);





	// 26-hood

	// Now the four corners of the bottom plane
	neighborhood[22] = (connectivity > 18) * (x > 0 && y > 0 && z < sz - 1) * (-1 - sx + sxy);
	neighborhood[23] = (connectivity > 18) * (x < sx - 1 && y > 0 && z < sz - 1) * (1 - sx + sxy);
	neighborhood[24] = (connectivity > 18) * (x > 0 && y < sy - 1 && z < sz - 1) * (-1 + sx + sxy);
	neighborhood[25] = (connectivity > 18) * (x < sx - 1 && y < sy - 1 && z < sz - 1) * (1 + sx + sxy);
}



const std::unordered_map<std::pair<int, int>, float, pair_hash, pair_equal>
extract_region_graph(
	int* labels,
	const int64_t sx, const int64_t sy, const int64_t sz,
	const float wx, const float wy, const float wz,
	const int64_t connectivity,
	const bool surface_area
) {

	const int64_t sxy = sx * sy;

	int neighborhood[13];
	float areas[13]; // all zero except faces

	if (surface_area) {
		for (int i = 3; i < 13; i++) {
			areas[i] = 0;
		}
		areas[0] = wy * wz; // x axis
		areas[1] = wx * wz; // y axis
		areas[2] = wx * wy; // z axis
	}
	else { // voxel counts
		for (int i = 0; i < 13; i++) {
			areas[i] = 1;
		}
	}

	int cur = 0;
	int label = 0;

	std::unordered_map<std::pair<int, int>, float, pair_hash, pair_equal> edges;

	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (cur == 0) {
					continue;
				}

				compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

				for (int i = 0; i < connectivity / 2; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];

					if (label == 0) {
						continue;
					}
					else if (cur > label) {
						edges[std::pair<int, int>(label, cur)] += areas[i];
					}
					else if (cur < label) {
						edges[std::pair<int, int>(cur, label)] += areas[i];
					}
				}
			}
		}
	}

	return edges;
}



int* _erode_ml(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity
) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sxy * sz;
	int neighborhood[26];

	int cur = 0;
	int label = 0;
	int m = 0;
	int* out_labels = new int[voxels]();
	bool is_bound;


	for (int64_t z = 1; z < sz-1; z++) {
		for (int64_t y = 1; y < sy-1; y++) {
			for (int64_t x = 1; x < sx-1; x++) {
				int64_t loc = x + sx * y + sxy * z;

				cur = labels[loc];

				if (cur == 0) {
					continue;
				}

				compute_neighborhood_all(neighborhood, x, y, z, sx, sy, sz, connectivity);
				out_labels[loc] = cur;
				for (int i = 0; i < connectivity; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];
					if (cur != label) {
						out_labels[loc] = 0;
					}
				}
				
			}
		}
	}

	return out_labels;
}



int* erode_ml_c(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity, const int64_t it) {


	int64_t all = sx * sy * sz;
	int* erode_labels = NULL;
	int* erode_labels_pre = labels;
	for (int i = 0; i < it; i++) {
		if (i > 0) {
			erode_labels_pre = erode_labels;
		}
		erode_labels = _erode_ml(erode_labels_pre, mask, sz, sy, sx, connectivity);
		if (erode_labels_pre != NULL && i > 0) {
			delete[] erode_labels_pre;
		}
	}
	return erode_labels;
}


int* _dilate_ml(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity
) {

	const int64_t sxy = sx * sy;
	const int64_t voxels = sxy * sz;
	int neighborhood[26];

	int cur = 0;
	int label = 0;

	int* out_labels = new int[voxels]();


	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (mask[loc] == 1) {
					continue;
				}

				if (cur != 0) {
					out_labels[loc] = cur;
					continue;
				}

				compute_neighborhood_all(neighborhood, x, y, z, sx, sy, sz, connectivity);

				for (int i = 0; i < connectivity; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					//if (neighboridx > voxels) {
					//	std::cout << x << "," << y << "," << z << std::endl;
					//	std::cout << neighborhood[i] << std::endl;
					//}
					label = labels[neighboridx];

					if (label != 0 && mask[neighboridx]!=1) {
						out_labels[loc] = label;
						break;
					}

					
				}
			}
		}
	}

	return out_labels;
}


int* dilate_ml_c(
	int* labels,
	int* mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity, const int64_t it
) {
	int64_t all = sx * sy * sz;
	int* dilate_labels = NULL;
	int* dilate_labels_pre = labels;
	for (int i = 0; i < it; i++) {
		if (i > 0) {
			dilate_labels_pre = dilate_labels;
		}
		dilate_labels = _dilate_ml(dilate_labels_pre, mask, sz, sy, sx);
		if (dilate_labels_pre != NULL && i > 0) {
			delete[] dilate_labels_pre;
		}
	}
	return dilate_labels;
}



const std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> extract_region_graph_background(
	int* labels,
	int* inner_mask,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity
) {

	const int64_t sxy = sx * sy;

	int neighborhood[13];
	int areas[13]; // all zero except faces

	// voxel counts
	for (int i = 0; i < 13; i++) {
		areas[i] = 1;
	}


	int cur = 0;
	int label = 0;

	std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> edges;
	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (cur == 0) {
					continue;
				}
				compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

				for (int i = 0; i < connectivity / 2; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];
					if (label > 0 && label != cur) {
						if (inner_mask[loc] == 0 && inner_mask[neighboridx] == 0) {
							if (cur > label) {
								edges[std::pair<int, int>(label, cur)].first += areas[i];
							}
							else if (cur < label) {
								edges[std::pair<int, int>(cur, label)].first += areas[i];
							}
						}
						else
						{
							if (cur > label) {
								edges[std::pair<int, int>(label, cur)].second += areas[i];
							}
							else if (cur < label) {
								edges[std::pair<int, int>(cur, label)].second += areas[i];
							}
						}
					}

				}
			}
		}
	}

	return edges;
}


const std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> extract_region_graph_cross(
	int* labels,
	int* sign,
	const int64_t sz, const int64_t sy, const int64_t sx,
	const int64_t connectivity
) {

	const int64_t sxy = sx * sy;

	int neighborhood[13];
	int areas[13]; // all zero except faces

	// voxel counts
	for (int i = 0; i < 13; i++) {
		areas[i] = 1;
	}


	int cur = 0;
	int label = 0;

	std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash, pair_equal> edges;
	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				int64_t loc = x + sx * y + sxy * z;
				cur = labels[loc];

				if (cur == 0) {
					continue;
				}
				compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

				for (int i = 0; i < connectivity / 2; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = labels[neighboridx];
					if (label > 0 && label != cur) {
						int a = sign[loc];
						int b = sign[neighboridx];
						if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
							if (cur > label) {
								edges[std::pair<int, int>(label, cur)].first += areas[i];
							}
							else if (cur < label) {
								edges[std::pair<int, int>(cur, label)].first += areas[i];
							}
						}
						else
						{
							if (cur > label) {
								edges[std::pair<int, int>(label, cur)].second += areas[i];
							}
							else if (cur < label) {
								edges[std::pair<int, int>(cur, label)].second += areas[i];
							}
						}
					}

				}
			}
		}
	}

	return edges;
}


bool detect_bound(int* input, size_t sz, size_t sy, size_t sx){
	size_t sxy = sx * sy;
	int connectivity = 26;
	int neighborhood[26];
	int label, record_label;

	for (int z = 0; z < sz; z++) {
		for (int y = 0; y < sy; y++) {
			for (int x = 0; x < sx; x++) {
				int loc = x + sx * y + sxy * z;

				int cur = input[loc];
				if (cur != 0) {
					continue;
				}
				record_label = 0;
				compute_neighborhood_all(neighborhood, x, y, z, sx, sy, sz, connectivity);
				for (int i = 0; i < connectivity; i++) {
					int64_t neighboridx = loc + neighborhood[i];
					label = input[neighboridx];
					if (record_label == 0) {
						record_label = label;
					}
					else
					{
						if (label != 0 && label != record_label) {
							std::cout << record_label << " and " << label << std::endl;
							return true;
						}
					}
				}
			}
		}
	}
	return false;
}

