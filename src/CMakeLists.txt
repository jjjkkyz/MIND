




add_subdirectory(mldf_cxx)
add_subdirectory(third/polyscope)
# 将源代码添加到此项目的可执行文件。
add_executable(WN_SEP "wn_sep.cpp")
target_link_libraries(WN_SEP PRIVATE WN_API)
target_link_libraries(WN_SEP PRIVATE polyscope)
set_property(TARGET WN_SEP PROPERTY CXX_STANDARD 20)
target_compile_features(WN_SEP PRIVATE cxx_std_20)
# 为目标添加额外的 MSVC 选项
if (MSVC)
    target_compile_options(WN_SEP PRIVATE
        /std:c++20    # 显式指定 C++20
        /permissive-  # 禁用微软特有扩展
        /W4           # 高级别警告
    )
endif()
# TODO: 如有需要，请添加测试并安装目标。
