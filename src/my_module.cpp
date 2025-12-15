#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // 引入 NumPy 支持
#include <pybind11/stl.h>

#include "mind_cxx/mldf_api.h"

namespace py = pybind11;


py::array_t<int> erode(py::array_t<int> input_array, py::array_t<int> mask_array, int i, int connectivity) {
    // 验证数组维度和类型
    if (input_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!input_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be double");
    }
    // 验证数组维度和类型
    if (mask_array.ndim() != 3) {
        throw std::runtime_error("Mask must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!mask_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Mask Data type must be double");
    }

    // 获取数组信息
    auto info = input_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)



    // 获取数组信息
    auto info_mask = mask_array.request();
    auto* mask = static_cast<int*>(info_mask.ptr);
    auto shape_mask = info_mask.shape; // 形状为 (depth, height, width)

    if (shape_mask[0] != shape[0] || shape_mask[1] != shape[1] || shape_mask[2] != shape[2]) {
        throw std::runtime_error("Input and Mask Must have same dim");
    }
    int* output_pt = erode_ml_c(data, mask, shape[0], shape[1], shape[2], connectivity, i);
    int64_t dim[3] = { info.shape[0],info.shape[1] ,info.shape[2] };
    //show<int>(output_pt, dim);
    // 创建 py::array_t，并绑定内存释放函数
    auto capsule = py::capsule(output_pt, [](void* p) {
        delete[] static_cast<int*>(p);
        });
    return py::array_t<int>(
        shape,        // 形状
        info.strides,      // 步长（可选，C 连续可省略）
        output_pt,         // 数据指针
        capsule       // 内存管理
    );
}


py::array_t<int> dilate(py::array_t<int> input_array, py::array_t<int> mask_array, int i, int connectivity) {
    // 验证数组维度和类型
    if (input_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!input_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be double");
    }
    // 验证数组维度和类型
    if (mask_array.ndim() != 3) {
        throw std::runtime_error("Mask must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!mask_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Mask Data type must be double");
    }

    // 获取数组信息
    auto info = input_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)



    // 获取数组信息
    auto info_mask = mask_array.request();
    auto* mask = static_cast<int*>(info_mask.ptr);
    auto shape_mask = info_mask.shape; // 形状为 (depth, height, width)

    if (shape_mask[0] != shape[0] || shape_mask[1] != shape[1] || shape_mask[2] != shape[2]) {
        throw std::runtime_error("Input and Mask Must have same dim");
    }
    int* output_pt = dilate_ml_c(data, mask, shape[0], shape[1], shape[2], connectivity, i);
    int64_t dim[3] = { info.shape[0],info.shape[1] ,info.shape[2] };

    // 创建 py::array_t，并绑定内存释放函数
    auto capsule = py::capsule(output_pt, [](void* p) {
        delete[] static_cast<int*>(p);
        });
    return py::array_t<int>(
        shape,        // 形状
        info.strides,      // 步长（可选，C 连续可省略）
        output_pt,         // 数据指针
        capsule       // 内存管理
    );
}


py::array_t<int> grid_cut(py::array_t<int> origin_array, py::array_t<int> relabel_array, py::array_t<float> udf_array, int N, int loop) {
    // 验证数组维度和类型
    if (origin_array.ndim() != 3 || relabel_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!origin_array.dtype().is(py::dtype::of<int>()) || !relabel_array.dtype().is(py::dtype::of<int>()) ) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = origin_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_re = relabel_array.request();
    auto* relabel = static_cast<int*>(info_re.ptr);
    auto shape_re = info_re.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_udf = udf_array.request();
    auto* udf_data = static_cast<float*>(info_udf.ptr);

    if (shape_re[0] != shape[0] || shape_re[1] != shape[1] || shape_re[2] != shape[2]) {
        throw std::runtime_error("Inputs Must have same dim");
    }
    auto out_array = py::array_t<int>(shape);
    int* output_pt = static_cast<int*>(out_array.request().ptr);
    std::memset(output_pt, 0, shape[0] * shape[1] * shape[2] * sizeof(int));
    try
    {
        grid_graph_cut(data,relabel, udf_data, output_pt, N, shape[0], shape[1], shape[2], loop);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return out_array;
}


py::array_t<int> grid_expansion(py::array_t<int> orient_array, py::array_t<int> unorient_array, py::array_t<int> sign, int N) {
    // 验证数组维度和类型
    if (orient_array.ndim() != 3 || unorient_array.ndim() != 3 || sign.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!orient_array.dtype().is(py::dtype::of<int>()) || !unorient_array.dtype().is(py::dtype::of<int>()) || !sign.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = orient_array.request();
    auto* orient_region = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_re = unorient_array.request();
    auto* unorient_region = static_cast<int*>(info_re.ptr);
    auto shape_re = info_re.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_sign = sign.request();
    auto* sign_data = static_cast<int*>(info_sign.ptr);
    auto shape_m2 = info_sign.shape; // 形状为 (depth, height, width)

    if (shape_re[0] != shape[0] || shape_re[1] != shape[1] || shape_re[2] != shape[2]) {
        throw std::runtime_error("Inputs Must have same dim");
    }
    auto out_array = py::array_t<int>(shape);
    int* output_pt = static_cast<int*>(out_array.request().ptr);
    std::memset(output_pt, 0, shape[0] * shape[1] * shape[2] * sizeof(int));
    grid_graph_expansion(orient_region, unorient_region, sign_data, output_pt, N, shape[0], shape[1], shape[2]);
    return out_array;
}


std::unordered_map<int, int> label_graph(py::array_t<int> origin_array, py::array_t<int> mask2, int re_N, float t) {
    // 验证数组维度和类型
    if (origin_array.ndim() != 3 || mask2.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!origin_array.dtype().is(py::dtype::of<int>()) || !mask2.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = origin_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_m2 = mask2.request();
    auto* mask2_data = static_cast<int*>(info_m2.ptr);
    auto shape_m2 = info_m2.shape; // 形状为 (depth, height, width)

    if (shape_m2[0] != shape[0] || shape_m2[1] != shape[1] || shape_m2[2] != shape[2]) {
        throw std::runtime_error("Inputs Must have same dim");
    }
    auto surface_contact = extract_region_graph_background(data, mask2_data, shape_m2[0], shape_m2[1], shape_m2[2]);
    auto label_map = label_merge(surface_contact, re_N, t);
    return label_map;
}


std::unordered_map<int, int> label_graph_merge(py::array_t<int> origin_array, py::array_t<int> sign, int re_N, float t) {
    // 验证数组维度和类型
    if (origin_array.ndim() != 3 || sign.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!origin_array.dtype().is(py::dtype::of<int>()) || !sign.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = origin_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_sign = sign.request();
    auto* sign_data = static_cast<int*>(info_sign.ptr);
    auto shape_m2 = info_sign.shape; // 形状为 (depth, height, width)

    if (shape_m2[0] != shape[0] || shape_m2[1] != shape[1] || shape_m2[2] != shape[2]) {
        throw std::runtime_error("Inputs Must have same dim");
    }
    auto surface_contact = extract_region_graph_cross(data, sign_data, shape_m2[0], shape_m2[1], shape_m2[2],6);
    auto label_map = label_merge(surface_contact, re_N,t);

    return label_map;
}


py::array_t<int> inplace_label_py (py::array_t<int> origin_array, std::unordered_map<int, int> label_map) {
    // 验证数组维度和类型
    if (origin_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!origin_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = origin_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)
    auto out_array = py::array_t<int>(shape);
    int* output_pt = static_cast<int*>(out_array.request().ptr);
    inplace_label(data, label_map, shape[0], shape[1], shape[2], output_pt);
    re_sort(output_pt, shape[0], shape[1], shape[2]);
    return out_array;
}


bool is_bound(py::array_t<int> origin_array) {
    // 验证数组维度和类型
    if (origin_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!origin_array.dtype().is(py::dtype::of<int>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = origin_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    return detect_bound(data, shape[0], shape[1], shape[2]);
}


py::tuple
m3c_py(py::array_t<int> label_array, py::array_t<double> value_array) {
    // 验证数组维度和类型
    if (label_array.ndim() != 3 || value_array.ndim() != 3) {
        throw std::runtime_error("Input must be a 3D array");
    }
    // 修改: 使用 !is() 替代 !=
    if (!label_array.dtype().is(py::dtype::of<int>()) || !value_array.dtype().is(py::dtype::of<double>())) {
        throw std::runtime_error("Input Data type must be int");
    }

    // 获取数组信息
    auto info = label_array.request();
    auto* data = static_cast<int*>(info.ptr);
    auto shape = info.shape; // 形状为 (depth, height, width)

    // 获取数组信息
    auto info_v = value_array.request();
    auto* value_data = static_cast<double*>(info_v.ptr);
    auto shape_v = info_v.shape; // 形状为 (depth, height, width)

    if (shape_v[0] != shape[0] || shape_v[1] != shape[1] || shape_v[2] != shape[2]) {
        throw std::runtime_error("Inputs Must have same dim");
    }
    // 保留了之前你做的 narrowing cast 修复
    size_t dim[3] = {static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1]), static_cast<size_t>(shape[2])};
    M3CEntireVolume m3c(data, value_data, dim);
    std::cout<< "begin m3c"<<std::endl;
    m3c.execute();

    py::ssize_t v_len = m3c.m_vetices.size();
    py::ssize_t v_c = 3;
    // 创建顶点数组
    py::array_t<float> vertices({v_len,v_c});

    auto result_buf = vertices.mutable_data();

    for (size_t i = 0; i < v_len; ++i) {
        for (size_t j = 0; j < v_c; ++j) {
            result_buf[i * v_c + j] = m3c.m_vetices[i][j];
        }
    }
    py::ssize_t f_len = m3c.m_faces.size();
    py::ssize_t f_c = 3;
    // 创建面片数组
    py::array_t<int> faces({ f_len, f_c });

    int* result_buf_face = faces.mutable_data();
    for (size_t i = 0; i < f_len; ++i) {
        for (size_t j = 0; j < f_c; ++j) {
            result_buf_face[i * f_c + j] = m3c.m_faces[i][j];
        }
    }

    py::ssize_t f_label_len = m3c.m_face_label.size();
    py::ssize_t f_label_c = 2;
    // 创建面片数组
    py::array_t<int> face_label({ f_label_len, f_label_c });

    int* result_buf_label = face_label.mutable_data();
    for (size_t i = 0; i < f_label_len; ++i) {
        result_buf_label[i * f_label_c] = m3c.m_face_label[i].first;
        result_buf_label[i * f_label_c + 1] = m3c.m_face_label[i].second;
    }

    return py::make_tuple(vertices, faces, face_label);
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "MLDF for UDF mesh extraction";
    m.def("erode", &erode, "Erode a 3d array given mask and iteration");
    m.def("dilate", &dilate, "Dilate a 3d array given mask and iteration");
    m.def("grid_cut", &grid_cut, "Grid cut a voxelgrid with alpha-expansion");
    m.def("grid_expansion", &grid_expansion, "Expansion the unorient region based on orient region");
    m.def("label_graph", &label_graph, "Extract label connectivity graph");
    m.def("label_graph_merge", &label_graph_merge, "Merge label cross block");
    m.def("inplace_label", &inplace_label_py, "Inplace label in 3d arrat given a label map");
    m.def("is_bound", &is_bound, "Detecting boundary block");
    m.def("m3c_py", &m3c_py, "M3C py api");
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}