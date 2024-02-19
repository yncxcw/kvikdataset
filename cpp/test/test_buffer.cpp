#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>

#include <buffer.hpp>

using namespace kvikdataset;

TEST(TarBufferTest, BufferConstruction)
{
  CPUBuffer<char> cpu_buffer(100);
  EXPECT_EQ(cpu_buffer.size(), 100);
  GPUBuffer<char> gpu_buffer(100);
  EXPECT_EQ(gpu_buffer.size(), 100);
}

TEST(TarBufferTest, BufferCPUMove)
{
  CPUBuffer<char> cpu_buffer(100);
  CPUBuffer<char> moved_buffer = std::move(cpu_buffer);
  EXPECT_TRUE(cpu_buffer.data() == nullptr);
  EXPECT_TRUE(moved_buffer.data() != nullptr);
}

TEST(TarBufferTest, BufferGPUMove)
{
  GPUBuffer<char> gpu_buffer(100);
  GPUBuffer<char> moved_buffer = std::move(gpu_buffer);
  EXPECT_TRUE(gpu_buffer.data() == nullptr);
  EXPECT_TRUE(moved_buffer.data() != nullptr);
}