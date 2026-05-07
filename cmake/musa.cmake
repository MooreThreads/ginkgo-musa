# cmake/musa.cmake - MUSA SDK discovery and compilation support for Ginkgo

find_path(MUSA_TOOLKIT_ROOT
    NAMES include/musa.h
    PATHS ENV MUSA_HOME /usr/local/musa
    DOC "MUSA SDK root directory"
)

if(NOT MUSA_TOOLKIT_ROOT)
    message(FATAL_ERROR "MUSA SDK not found. Set MUSA_HOME.")
endif()

set(MUSA_INCLUDE_DIRS "${MUSA_TOOLKIT_ROOT}/include")

find_program(MUSA_MCC_EXECUTABLE NAMES mcc
    PATHS "${MUSA_TOOLKIT_ROOT}/bin" DOC "MUSA mcc compiler")

find_library(MUSA_RUNTIME_LIBRARY NAMES musart PATHS "${MUSA_TOOLKIT_ROOT}/lib")
find_library(MUSA_MUBLAS_LIBRARY NAMES mublas PATHS "${MUSA_TOOLKIT_ROOT}/lib")
find_library(MUSA_MUSPARSE_LIBRARY NAMES musparse PATHS "${MUSA_TOOLKIT_ROOT}/lib")
find_library(MUSA_MURAND_LIBRARY NAMES murand PATHS "${MUSA_TOOLKIT_ROOT}/lib")
find_library(MUSA_MUFFT_LIBRARY NAMES mufft PATHS "${MUSA_TOOLKIT_ROOT}/lib")

set(MUSA_LIBRARIES
    ${MUSA_RUNTIME_LIBRARY}
    ${MUSA_MUBLAS_LIBRARY}
    ${MUSA_MUSPARSE_LIBRARY}
    ${MUSA_MURAND_LIBRARY}
    ${MUSA_MUFFT_LIBRARY}
)

if(NOT MUSA_ARCH)
    set(MUSA_ARCH "22" CACHE STRING "MUSA GPU arch (22=S4000, 31=S5000)")
endif()

message(STATUS "MUSA SDK: ${MUSA_TOOLKIT_ROOT}")
message(STATUS "MUSA mcc: ${MUSA_MCC_EXECUTABLE}")
message(STATUS "MUSA arch: mp_${MUSA_ARCH}")

# Function: compile a list of source files with mcc.
# .mu files: compiled directly with -mtgpu
# .cpp files: compiled with -mtgpu -x cuda (device code)
# Plain .cpp (host-only): compiled with g++ (no device code)
# Output: sets ${out_var} to list of object files in parent scope.
function(musa_compile out_var)
    set(_objects "")
    set(_base_flags
        -mtgpu --cuda-gpu-arch=mp_${MUSA_ARCH}
        -std=c++17 -O2 -fPIC
        -DGKO_COMPILING_MUSA -DGKO_DEVICE_NAMESPACE=musa
        -I${MUSA_INCLUDE_DIRS}
        -I${PROJECT_SOURCE_DIR}
        -I${PROJECT_SOURCE_DIR}/include
        -I${PROJECT_BINARY_DIR}/include
    )
    foreach(_src ${ARGN})
        # Make absolute
        if(NOT IS_ABSOLUTE "${_src}")
            set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        endif()
        # Unique object name
        file(RELATIVE_PATH _rel "${PROJECT_SOURCE_DIR}" "${_src}")
        string(REPLACE "/" "_" _obj_name "${_rel}")
        string(REGEX REPLACE "\\.[^.]+$" ".o" _obj_name "${_obj_name}")
        set(_obj "${CMAKE_CURRENT_BINARY_DIR}/musa_objs/${_obj_name}")

        get_filename_component(_ext "${_src}" EXT)
        set(_extra_flags "")
        if(_ext STREQUAL ".cpp")
            set(_extra_flags -x cuda)
        endif()

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/musa_objs"
            COMMAND ${MUSA_MCC_EXECUTABLE}
                ${_base_flags} ${_extra_flags}
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -I${CMAKE_CURRENT_BINARY_DIR}
                -c "${_src}" -o "${_obj}"
            DEPENDS "${_src}"
            COMMENT "MUSA: ${_rel}"
            VERBATIM
        )
        list(APPEND _objects "${_obj}")
    endforeach()
    set(${out_var} ${_objects} PARENT_SCOPE)
endfunction()
