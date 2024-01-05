#include <gtest/gtest.h>
#include <tar_reader.hpp>

TEST(TarReaderTest, TarReaderConstruction)
{
  kvikdataset::TarReader reader("../../cpp/test/test_data/test.tar");
}