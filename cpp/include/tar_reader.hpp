/*
 * =============================================================================
 * Copyright (c) 2023, ynjassionchen@gmail.com
 * =============================================================================
 */
#pragma once

#include <buffer.hpp>
#include <fstream>
#include <string>
#include <unordered_map>

namespace kvikdataset {

static const std::unordered_map<std::string, std::tuple<size_t, size_t>> TarMetadata = {
  {"name", std::make_tuple(0, 100)},
  {"size", std::make_tuple(124, 12)},
  {"checksum", std::make_tuple(148, 12)},
};

// Device to place the content of each archive.
enum class ArchiveDevice { CPU, GPU };

struct TarArchive {
  // Name of the archive.
  std::string name;
  // The size of the archive.
  size_t size;
  // Deice to place the content of the archive
  ArchiveDevice device;
  // Buffer for hosting the data on GPU.
  std::unique_ptr<GPUBuffer<char>> gpu_buffer;
  // Buffer for hosting the data on CPU.
  std::unique_ptr<CPUBuffer<char>> cpu_buffer;

  TarArchive(const std::string& name, const size_t& size, const ArchiveDevice& device)
    : name(name), size(size), device(device)
  {
  }

  TarArchive(const TarArchive& archive)
    : name(archive.name), size(archive.size), device(archive.device)
  {
    if (gpu_buffer != nullptr or cpu_buffer != nullptr) {
      throw std::runtime_error("Copying TarArchive with cpu/gpu buffer.");
    }
  }
};

class TarReader {
 public:
  TarReader(const std::string& file_path) : file_path(file_path)
  {
    // We only read the header to decode the archive.
    std::ifstream fin(file_path, std::ios_base::in | std::ios_base::binary);
    std::array<char, 512> header;

    if (!fin.is_open()) { throw std::runtime_error("file " + file_path + " can't be open."); }

    while (fin) {
      fin.read(header.data(), 512);
      if (std::size(header) != 512) {
        throw std::runtime_error("file " + file_path + " head read is failed.");
      }
      // Decoder header and populate the archives.
      const auto archive = decoder_header(header);
      archives.emplace(std::make_pair(archive.name, archive));
      long int offset = static_cast<long int>(archive.size + 512);
      fin.seekg(fin.tellg() + offset);
    }
  }

  TarReader(const TarReader& tar_reader) = delete;

  TarReader(TarReader&& tar_reader) = delete;

  TarReader& operator=(TarReader& tar_reader) = delete;

  void read() {}

 private:
  TarArchive decoder_header(const std::array<char, 512>& header)
  {
    const auto name = retrive_header_field(header, "name");
    const auto size = std::stoi(retrive_header_field(header, "size"));

    ArchiveDevice device;
    if (name.substr(name.size() - 4) == "_cpu") {
      device = ArchiveDevice::CPU;
    } else if (name.substr(name.size() - 4) == "_gpu") {
      device = ArchiveDevice::GPU;
    } else {
      throw std::runtime_error("Invalid archive name " + name);
    }
    return TarArchive(name, size, device);
  }

  const std::string retrive_header_field(const std::array<char, 512> header,
                                         const std::string& field) const
  {
    const auto& offset = std::get<0>(TarMetadata.at(field));
    const auto& len    = std::get<1>(TarMetadata.at(field));
    return std::string(header.begin() + offset, header.begin() + offset + len);
  }

  std::string file_path;
  std::unordered_map<std::string, TarArchive> archives;
};

}  // namespace kvikdataset