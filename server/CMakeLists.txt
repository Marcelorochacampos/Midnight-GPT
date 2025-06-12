cmake_minimum_required(VERSION 3.31)
project(torch_api)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH /Users/marcelocampos/Desktop/projects/midnight-gpt/server/third_party/libtorch)

find_package(Torch REQUIRED)

include_directories(third_party/boost)
include_directories(third_party/asio-1.30.2/include)
include_directories(third_party/crow/include)
include_directories(third_party/json/single_include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h")

add_executable(server ${SOURCES} ${HEADERS})
target_link_libraries(server "${TORCH_LIBRARIES}")

set_property(TARGET server PROPERTY CXX_STANDARD 20)

