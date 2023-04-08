#include <nvbench/nvbench.hpp>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.hpp>

void simple(nvbench::state &state)
{
  state.exec([](nvbench::launch &launch) {
    // Sleep for 1 millisecond:
    nvbench::sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(1e-3);
  });
}
NVBENCH_BENCH(simple);
