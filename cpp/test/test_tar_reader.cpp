#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>

#include <kvikio/kvikio.hpp>
#include <tar_reader.hpp>

using namespace kvikdataset;

void _cuda_check(bool condition)
{
  if (!condition) {
    std::cout << "Error" << std::endl;
    exit(-1);
  }
}

template <class T>
void _buffer_compare_cpu(void* buffer, size_t size, const T v)
{
  EXPECT_TRUE(kvikio::is_host_memory(buffer));
  T* cast_ptr = static_cast<T*>(buffer);
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(cast_ptr[i], v);
  }
}

template <class T>
void _buffer_compare_gpu(void* buffer, size_t size, T v)
{
  EXPECT_FALSE(kvikio::is_host_memory(buffer));
  T* host_ptr = new T[size];
  _cuda_check(cudaMemcpy(host_ptr, buffer, size, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(host_ptr[i], v);
  }
  delete host_ptr;
}

TEST(TarReaderTest, TarReaderConstruction)
{
  TarReader reader("../../cpp/test/test_data/test.tar");
  auto& archives = reader.read_archives();
  {
    EXPECT_EQ(archives["foo_cpu"].name(), "foo_cpu");
    EXPECT_EQ(archives["foo_cpu"].size(), 100);
    EXPECT_EQ(archives["foo_cpu"].device(), ArchiveDevice::CPU);
  }
  {
    EXPECT_EQ(archives["boo_gpu"].name(), "boo_gpu");
    EXPECT_EQ(archives["boo_gpu"].size(), 200);
    EXPECT_EQ(archives["boo_gpu"].device(), ArchiveDevice::GPU);
  }
  {
    EXPECT_EQ(archives["poo_gpu"].name(), "poo_gpu");
    EXPECT_EQ(archives["poo_gpu"].size(), 300);
    EXPECT_EQ(archives["poo_gpu"].device(), ArchiveDevice::GPU);
  }
  {
    EXPECT_EQ(archives.size(), 3);
  }
}

TEST(TarReaderTest, TarReaderRead)
{
  TarReader reader("../../cpp/test/test_data/test.tar");
  reader.read();
  auto& archives = reader.read_archives();
  for (auto& archive : archives) {
    EXPECT_EQ(archive.second.is_read(), true);
    if (archive.first == "foo_cpu") {
      EXPECT_TRUE(archive.second.device() == ArchiveDevice::CPU);
      _buffer_compare_cpu<uint8_t>(
        archive.second.cpu_buffer()->data(), archive.second.size(), static_cast<uint8_t>(2));
    }

    if (archive.first == "boo_gpu") {
      EXPECT_TRUE(archive.second.device() == ArchiveDevice::GPU);
      _buffer_compare_gpu<uint8_t>(archive.second.gpu_buffer()->data(), archive.second.size(), 4);
    }

    if (archive.first == "foo_cpu") {
      EXPECT_TRUE(archive.second.device() == ArchiveDevice::GPU);
      _buffer_compare_gpu<uint8_t>(archive.second.gpu_buffer()->data(), archive.second.size(), 6);
    }
  }
}