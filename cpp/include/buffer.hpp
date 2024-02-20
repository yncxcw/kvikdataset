/*
 * =============================================================================
 * Copyright (c) 2023, ynjassionchen@gmail.com
 * =============================================================================
 */
#pragma once

#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

namespace kvikdataset {

class Allocator {
 public:
  virtual bool malloc(void** ptr, const size_t size) const = 0;
  virtual void free(void* ptr) const                       = 0;
  virtual std::string type() const                         = 0;
};

class CPUAllocator : public Allocator {
 public:
  bool malloc(void** ptr, const size_t size) const
  {
    if (*ptr != nullptr) { throw std::runtime_error("malloc on non-nullptr."); }
    *ptr = std::malloc(size);
    std::memset(*ptr, 0, size);
    return *ptr != nullptr;
  }

  void free(void* ptr) const
  {
    if (ptr != nullptr) std::free(ptr);
  }

  std::string type() const { return "cpu"; }
};

class GPUAllocator : public Allocator {
 public:
  bool malloc(void** ptr, const size_t size) const
  {
    if (*ptr != nullptr) { throw std::runtime_error("cudaMalloc on non-nullptr."); }
    if (cudaMalloc(ptr, size) != cudaSuccess) {
      throw std::runtime_error("Failed memory allocation on GPU.");
    }
    return true;
  }

  void free(void* ptr) const
  {
    if (ptr != nullptr) cudaFree(ptr);
  }

  std::string type() const { return "gpu"; }
};

template <class MemAllocator, class T>
class Buffer {
 public:
  Buffer() : _size(0), _buffer(nullptr) {}

  Buffer(size_t size)
  {
    _size = size;
    if (!allocator.malloc((void**)(&_buffer), sizeof(T) * _size)) { throw std::bad_alloc(); }
    if (_buffer == nullptr) { throw std::bad_alloc(); }
  }

  Buffer(Buffer&& buffer) : _size(buffer._size), _buffer(buffer._buffer)
  {
    buffer._buffer = nullptr;
    buffer._size   = 0;
  }

  Buffer(const Buffer& buffer) = delete;

  Buffer& operator=(Buffer& buffer) = delete;

  ~Buffer()
  {
    allocator.free((void*)(_buffer));
    _size   = 0;
    _buffer = nullptr;
  }

  // For debugging purpose
  void print() const
  {
    if (std::is_same<MemAllocator, CPUAllocator>::value) {
      for (size_t i = 0; i < _size; i++) {
        std::cout << _buffer[i] - 0 << std::endl;
      }
    }
  }

  bool empty() const { return _buffer == nullptr; }

  void* data() const { return _buffer; }

  const size_t size() const { return _size; }

 private:
  // Size of the buffer.
  size_t _size;

  // Raw ptr of the buffer.
  T* _buffer{nullptr};

  MemAllocator allocator;
};

template <class T>
using GPUBuffer = Buffer<GPUAllocator, T>;

template <class T>
using CPUBuffer = Buffer<CPUAllocator, T>;

}  // namespace kvikdataset
