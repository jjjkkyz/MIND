add_subdirectory(gco-v3.0)
include_directories("connected-components-3d")
include_directories("gco-v3.0")



add_library (WN_API STATIC "mldf_api.cpp" "m3c.cpp" "utils.cpp" ${_GCO_SOURCES})
set_property(TARGET WN_API PROPERTY POSITION_INDEPENDENT_CODE ON)


set_property(TARGET WN_API PROPERTY CXX_STANDARD 20)

# TODO: ������Ҫ�������Ӳ��Բ���װĿ�ꡣ
