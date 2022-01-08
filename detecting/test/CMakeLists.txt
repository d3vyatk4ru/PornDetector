cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)

project(tests)

add_compile_options(-g -O0 -Wextra -Wall -Wextra -Wpedantic -lgcov --coverage)
add_link_options( -O0 -lgcov --coverage)

add_compile_options(-pthread)
list(APPEND CMAKE_PREFIX_PATH "/home/runner/work/BroBand/BroBand/detecting/libtorch")

include_directories("../include")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Включаем тестирование
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Исполняемый файл для динамической библиотеки и многопоточной реализации
add_executable(test_detect test.cpp "../src/detecting.cpp")

# Линкуем (соединяем) программу со статической библиотекой
target_link_libraries(test_detect ${GTEST_LIBRARIES} pthread "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
