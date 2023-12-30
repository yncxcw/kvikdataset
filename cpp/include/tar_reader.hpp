/*
 * =============================================================================
 * Copyright (c) 2023, ynjassionchen@gmail.com
 * =============================================================================
 */
#pragma once

#include <buffer.hpp>
#include <string>
#include <unordered_map>

namespace kvikdataset {

// Device to place the content of each archive.
enum class ArchiveDevice { CPU, GPU };

struct TarArchive {
  // Name of the archive.
  std::string archive_name;
  // The size of the archive.
  size_t size;
  // Deice to place the content of the archive
  ArchiveDevice device;
  // Buffer for hosting the data on GPU.
  std::unique_ptr<GPUBuffer<char>> gpu_buffer;
  // Buffer for hosting the data on CPU.
  std::unique_ptr<CPUBuffer<char>> cpu_buffer;
};

class TarReader {
 public:
  TarReader(const std::string& file_path) : file_path(file_path) {}

  void read() private : void init_tar() {}

  std::string file_path;
  std::unordered_map<std::string, TarArchive> archives;
};

}  // namespace kvikdataset