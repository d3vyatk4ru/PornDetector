# минимальная версия для запуска
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
# имя проекта
project(detecting_service)

add_compile_options(-pthread)
list(APPEND CMAKE_PREFIX_PATH "/home/runner/work/BroBand/BroBand/detecting/libtorch")

# добавляем поиск файлов в директории include
include_directories("../include")

find_package(Torch REQUIRED)
include_directories("../libtorch/include")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# библиотека с общими функциями
set(DETECT_LIB_SOURCE detecting.cpp)

# создание статической библиотеки
add_library(detect STATIC ${DETECT_LIB_SOURCE})

# создание бинарных файлов
add_executable(main_detect main.cpp detecting.cpp)

# соединение библиотеки и основного бинарного файла
target_link_libraries(main_detect m pthread ${TORCH_LIBRARIES} ${OpenCV_LIBS})
