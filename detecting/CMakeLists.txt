# минимальная версия для запуска
cmake_minimum_required(VERSION 3.16)
# имя проекта
project(BroBand_ML)

# стандарт
set(CMAKE_CXX_STANDARD 14)

# добавляем директории
add_subdirectory(src)
add_subdirectory(test)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)


