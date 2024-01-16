#include <gtest/gtest.h>
#include <tar_reader.hpp>

using namespace kvikdataset;

TEST(TarReaderTest, TarReaderConstruction)
{
  TarReader reader("../../cpp/test/test_data/test.tar");
  auto archives = reader.read_archives();
  {
    EXPECT_EQ(archives["foo_cpu"].name(), "foo_cpu");
    EXPECT_EQ(archives["foo_cpu"].size(), 16);
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
}