cmake_minimum_required(VERSION 3.20)

project(multiWindowSync C CXX)

# Common directories.
set(BIN_DIR  ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)

### Libraries
# Tracy

set(TRACY_ENABLE ON CACHE BOOL "")
set(TRACY_ONLY_IPV4 ON CACHE BOOL "") # Prevents some connection confusion because the server UI uses 127.0.0.1 by default.
set(TRACY_STATIC OFF CACHE BOOL "") # This is required since we are a multi-dll program.
set(TRACY_ON_DEMAND OFF CACHE BOOL "")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tracy")

# Build binaries and pdbs into our output bin/pdb directories.
set_target_properties(TracyClient
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if(WIN32)
  set_target_properties(TracyClient
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Agility SDK
add_subdirectory(DirectXAgilitySDK)

### Executable
set(exec_target multiWindowSync)

add_executable(${exec_target})
target_sources(${exec_target}
  PRIVATE
    src/main.cpp)

target_compile_definitions(${exec_target}
  PRIVATE
  TRACY_IMPORTS)

  
target_link_libraries(${exec_target}
  PRIVATE
  Tracy::TracyClient
  d3d12
  dxgi
  d3dcompiler
  agilitysdk
  agilitysdk_d3dx12)

# Copy Agility SDK DLLs to correct location & pass version to the target
# See https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/ for more info
add_custom_command(TARGET ${exec_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${exec_target}>/D3D12/"
        COMMAND ${CMAKE_COMMAND} -E copy        -t "$<TARGET_FILE_DIR:${exec_target}>/D3D12/" ${AGILITYSDK_D3D12CORE_DLL}
        COMMAND ${CMAKE_COMMAND} -E copy        -t "$<TARGET_FILE_DIR:${exec_target}>/D3D12/" ${AGILITYSDK_D3D12SDKLAYERS_DLL}
        COMMAND_EXPAND_LISTS
)
