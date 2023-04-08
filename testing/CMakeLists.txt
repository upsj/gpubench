set(test_srcs
  axes_metadata.gpu.cpp
  benchmark.gpu.cpp
  create.gpu.cpp
  cuda_timer.gpu.cpp
  cpu_timer.gpu.cpp
  enum_type_list.gpu.cpp
  float64_axis.gpu.cpp
  int64_axis.gpu.cpp
  named_values.gpu.cpp
  option_parser.gpu.cpp
  range.gpu.cpp
  ring_buffer.gpu.cpp
  runner.gpu.cpp
  state.gpu.cpp
  state_generator.gpu.cpp
  string_axis.gpu.cpp
  type_axis.gpu.cpp
  type_list.gpu.cpp
)

set_source_files_properties(${test_srcs} PROPERTIES LANGUAGE ${GPU_LANG})

# Metatarget for all examples:
add_custom_target(nvbench.test.all)
add_dependencies(nvbench.all nvbench.test.all)

foreach(test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WLE)
  string(PREPEND test_name "nvbench.test.")
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(${test_name} PRIVATE nvbench::nvbench fmt)
  set_target_properties(${test_name} PROPERTIES COMPILE_FEATURES cuda_std_17)
  nvbench_config_target(${test_name})
  add_test(NAME ${test_name} COMMAND "$<TARGET_FILE:${test_name}>")

  add_dependencies(nvbench.test.all ${test_name})
endforeach()

add_subdirectory(cmake)
add_subdirectory(device)
