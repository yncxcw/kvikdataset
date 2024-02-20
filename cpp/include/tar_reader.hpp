/*
 * =============================================================================
 * Copyright (c) 2023, ynjassionchen@gmail.com
 * =============================================================================
 */
#pragma once

#include <buffer.hpp>
#include <fstream>
#include <kvikio/kvikio.hpp>
#include <string>
#include <unordered_map>

namespace kvikdataset {

// The size of each archive block, each archive is consists of multiple blocks
static constexpr const long int TAR_ARCHIVE_BLOCK_SIZE = 512;
static constexpr const long int TAR_ARCHIVE_HEAD_SIZE  = 512;

// Mapping from fields of header to a tuple of offset and length of this field
static const std::unordered_map<std::string, std::tuple<size_t, size_t>> TarMetadata = {
  {"name", std::make_tuple(0, 100)},
  {"size", std::make_tuple(124, 12)},
  {"checksum", std::make_tuple(148, 12)},
};

// Device to place the content of each archive.
enum class ArchiveDevice { CPU, GPU };

class TarArchive {
 public:
  TarArchive() {}
  TarArchive(const std::string& name,
             const size_t& size,
             const size_t& offset,
             const ArchiveDevice& device)
    : _name(name), _size(size), _offset(offset), _device(device)
  {
    if (_device == ArchiveDevice::CPU)
      _cpu_buffer = std::make_unique<CPUBuffer<char>>(size);
    else
      _gpu_buffer = std::make_unique<GPUBuffer<char>>(size);
  }

  TarArchive(const TarArchive& archive)
    : _name(archive._name), _size(archive._size), _offset(archive._offset), _device(archive._device)
  {
    if (_device == ArchiveDevice::CPU)
      if (_gpu_buffer != nullptr or _cpu_buffer != nullptr) {
        throw std::runtime_error("Copying TarArchive with cpu/gpu buffer.");
      }
    if (_device == ArchiveDevice::CPU)
      _cpu_buffer = std::make_unique<CPUBuffer<char>>(_size);
    else
      _gpu_buffer = std::make_unique<GPUBuffer<char>>(_size);
  }

  TarArchive(TarArchive&& archive)
    : _name(archive._name), _size(archive._size), _offset(archive._offset), _device(archive._device)
  {
    if (archive._cpu_buffer != nullptr) {
      _cpu_buffer         = std::move(archive._cpu_buffer);
      archive._cpu_buffer = nullptr;
    }

    if (archive._gpu_buffer != nullptr) {
      _gpu_buffer         = std::move(archive._gpu_buffer);
      archive._gpu_buffer = nullptr;
    }
  }

  ~TarArchive() {}
  std::string name() const { return _name; }

  size_t size() const { return _size; }

  size_t offset() const { return _offset; }

  bool is_read() const { return _is_read; }

  ArchiveDevice device() const { return _device; }

  void set_read() { _is_read = true; }

  std::unique_ptr<GPUBuffer<char>>& gpu_buffer() { return _gpu_buffer; }

  std::unique_ptr<CPUBuffer<char>>& cpu_buffer() { return _cpu_buffer; }

 private:
  // Name of the archive.
  std::string _name;
  // The size of the archive.
  size_t _size;
  // The offset of the this archive in the file.
  size_t _offset;
  // Deice to place the content of the archive
  ArchiveDevice _device;
  // If the archive has been read.
  bool _is_read{false};
  // Buffer for hosting the data on GPU.
  std::unique_ptr<GPUBuffer<char>> _gpu_buffer{nullptr};
  // Buffer for hosting the data on CPU.
  std::unique_ptr<CPUBuffer<char>> _cpu_buffer{nullptr};
};

class TarReader {
 public:
  TarReader(const std::string& file_path) : file_path(file_path)
  {
    // We only read the header to decode the archive.
    std::ifstream fin(file_path, std::ios_base::in | std::ios_base::binary);
    std::array<char, TAR_ARCHIVE_HEAD_SIZE> header;
    static const std::array<char, TAR_ARCHIVE_HEAD_SIZE> null_buffer{};

    if (!fin.is_open()) { throw std::runtime_error("file " + file_path + " can't be open."); }

    size_t current_offset = 0;
    while (fin) {
      fin.read(header.data(), TAR_ARCHIVE_HEAD_SIZE);
      if (std::size(header) != TAR_ARCHIVE_HEAD_SIZE) {
        throw std::runtime_error("file " + file_path + " head read is failed.");
      }

      // At the end of the tar file there is 2 x TAR_ARCHIVE_HDEA_SIZE empty buffer
      if (header == null_buffer) {
        fin.read(header.data(), TAR_ARCHIVE_HEAD_SIZE);
        if (header == null_buffer) {
          break;
        } else {
          throw std::runtime_error("file " + file_path + " invalid training part.");
        }
      }

      // The header of the current tar archive.
      current_offset += TAR_ARCHIVE_HEAD_SIZE;
      // Decoder header and populate the archives.
      TarArchive archive{decoder_header(header, current_offset)};
      // Note: With emplace(), you can inter map with rvalue.
      // This moves the unique_ptr of cpu/gpu buffer to archives, which avoids an
      // extra memory copy.

      archives.emplace(std::make_pair(archive.name(), std::move(archive)));
      // offset must be multiple blocks of TAR_ARCHIVE_BLOCK_SIZE
      long int offset = static_cast<long int>(
        archive.size() + (TAR_ARCHIVE_BLOCK_SIZE - archive.size() % TAR_ARCHIVE_BLOCK_SIZE) %
                           TAR_ARCHIVE_BLOCK_SIZE);
      fin.seekg(fin.tellg() + offset);
      current_offset += offset;
    }
  }

  ~TarReader() {}
  TarReader(const TarReader& tar_reader) = delete;

  TarReader(TarReader&& tar_reader) = delete;

  TarReader& operator=(TarReader& tar_reader) = delete;

  bool read()
  {
    kvikio::FileHandle f(file_path, "r");
    // the future size and expected size pair.
    std::vector<std::pair<std::future<std::size_t>, std::size_t>> futures;
    for (auto& archive : archives) {
      if (archive.second.device() == ArchiveDevice::CPU) {
        futures.push_back(std::make_pair(f.pread(
                                           // Address to device or host memory.
                                           archive.second.cpu_buffer()->data(),
                                           // Size of bytes to load.
                                           archive.second.size(),
                                           // Offset in the file to read from.
                                           archive.second.offset()),
                                         archive.second.size()));
        archive.second.set_read();
      } else if (archive.second.device() == ArchiveDevice::GPU) {
        futures.push_back(std::make_pair(f.pread(
                                           // Address to device or host memory.
                                           archive.second.gpu_buffer()->data(),
                                           // Size of bytes to load.
                                           archive.second.size(),
                                           // Offset in the file to read from.
                                           archive.second.offset()),
                                         archive.second.size()));
        archive.second.set_read();
      } else {
        throw std::runtime_error("Device not supported");
      }
    }

    // Sync on all reads.
    for (auto& future : futures) {
      // Return false, if it false to read one archive.
      if (future.first.get() != future.second) { return false; }
    }
    return true;
  }

  std::unordered_map<std::string, TarArchive>& read_archives() { return archives; }

 private:
  TarArchive decoder_header(const std::array<char, TAR_ARCHIVE_HEAD_SIZE>& header,
                            const size_t& offset)
  {
    const auto name = retrive_header_field(header, "name");
    const auto size = std::stoi(retrive_header_field(header, "size"), nullptr, 8);

    ArchiveDevice device;
    if (name.substr(name.size() - 4) == "_cpu") {
      device = ArchiveDevice::CPU;
    } else if (name.substr(name.size() - 4) == "_gpu") {
      device = ArchiveDevice::GPU;
    } else {
      throw std::runtime_error("Invalid archive name " + name);
    }

    return TarArchive(name, size, offset, device);
  }

  std::string retrive_header_field(const std::array<char, TAR_ARCHIVE_HEAD_SIZE>& header,
                                   const std::string& field) const
  {
    const auto& offset = std::get<0>(TarMetadata.at(field));
    const auto& len    = std::get<1>(TarMetadata.at(field));
    std::string str(header.begin() + offset, header.begin() + offset + len);
    str.erase(str.find_first_of('\0'), std::string::npos);
    return str;
  }

  std::string file_path;
  std::unordered_map<std::string, TarArchive> archives;
};

}  // namespace kvikdataset