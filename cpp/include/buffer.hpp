/*
 * =============================================================================
 * Copyright (c) 2023, ynjassionchen@gmail.com
 * =============================================================================
 */
#pragma once

#include <cuda_runtime_api.h>
#include <memory>

namespace kvikdataset {

class Allocator {
  virtual bool malloc(void** ptr, const size_t size) const = 0;
  virtual void free(void* ptr) const                       = 0;
};

class CPUAllocator : public Allocator {
  bool malloc(void** ptr, const size_t size) const
  {
    *ptr = std::malloc(size);
    return *ptr != nullptr;
  }

  void free(void* ptr) const { free(ptr); }
};

class GPUAllocator : public Allocator {
  bool malloc(void** ptr, const size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }

  void free(void* ptr) const { cudaFree(ptr); }
};

template <class MemAllocator, class T>
class Buffer {
 public:
  Buffer() : _size(0), _buffer(nullptr) {}

  Buffer(size_t size)
  {
    _size = size;
    if (!allocator.malloc(&_buffer, sizeof(T) * _size)) { throw std::bad_alloc(); }
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
    allocator.free(_buffer);
    _size = 0;
  }

  bool empty() { return _buffer == nullptr; }

  const void* data() const { return _buffer; }

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
