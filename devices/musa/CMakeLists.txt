ginkgo_add_object_library(ginkgo_musa_device executor.cpp)
if(GINKGO_BUILD_MUSA)
    target_compile_definitions(
        ginkgo_musa_device
        PRIVATE GKO_COMPILING_MUSA_DEVICE
    )
endif()
