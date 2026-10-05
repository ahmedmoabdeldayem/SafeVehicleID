cmake_minimum_required(VERSION 3.16)
project(SafeVehicleID CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
    # Enable auto-vectorization reporting
    # add_compile_options(-fopt-info-vec-optimized)
else()
    add_compile_options(-O0 -g -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# ─── Dependencies ─────────────────────────────────────────────────────────────

# OpenCV — required
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Threads — required
find_package(Threads REQUIRED)

# ─── Source files ─────────────────────────────────────────────────────────────
set(SOURCES
    src/tracking/kalman_tracker.cpp
    src/tracking/sort_tracker.cpp
    src/pipeline/frame_pipeline.cpp
)

# ─── Library ──────────────────────────────────────────────────────────────────
add_library(safevid_lib STATIC ${SOURCES})

target_include_directories(safevid_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(safevid_lib
    PUBLIC
        ${OpenCV_LIBS}
        Threads::Threads
)

# ─── Main executable ──────────────────────────────────────────────────────────
add_executable(safe_vehicle_id src/main.cpp)
target_link_libraries(safe_vehicle_id PRIVATE safevid_lib)

# ─── Tests ────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(test_kalman
        tests/test_kalman.cpp
    )
    target_link_libraries(test_kalman PRIVATE safevid_lib GTest::gtest_main)
    add_test(NAME KalmanTests COMMAND test_kalman)

    add_executable(test_sort
        tests/test_sort.cpp
    )
    target_link_libraries(test_sort PRIVATE safevid_lib GTest::gtest_main)
    add_test(NAME SORTTests COMMAND test_sort)

    add_executable(test_ring_buffer
        tests/test_ring_buffer.cpp
    )
    target_link_libraries(test_ring_buffer PRIVATE safevid_lib GTest::gtest_main)
    add_test(NAME RingBufferTests COMMAND test_ring_buffer)
endif()

# ─── Install ──────────────────────────────────────────────────────────────────
install(TARGETS safe_vehicle_id DESTINATION bin)
