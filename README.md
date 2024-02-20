# kvikdataset

## Introduction
`kvikdataset` is a c++ library used to load tar file. It uses Nvidia kivikio library with the support of direct I/O support for fast GPU/CPU I/O.

The primary usecase of this libary is for large scale deep learning training where the training data is serialized into tar files (like Webdataset).

## Example
### Step 1: create a tar file
The following example shows how to craete tar file with 3 archives. Note if the nmae of an archive ends with `_cpu`/`_gpu`, it loads into CPU/GPU memory respectively. 
```python
def write_archive(tarsteam, size: int, name: str, value: int):
    array = np.full(size, value, dtype=np.uint8)
    print(array.size)
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.uname = "wei"
    print(len(array.tobytes()))
    tarstream.addfile(ti, io.BytesIO(array.tobytes()))

if __name__ == "__main__":
    tarstream = tarfile.open("test.tar", mode="w")
    write_archive(tarstream, 10, "foo_cpu", 2)
    write_archive(tarstream, 10, "boo_gpu", 4)
    write_archive(tarstream, 10, "poo_gpu", 6)
    tarstream.close()
```

### Step 3: loads with kvikdataset.
The following exampels shows how to laod tar file with kvikdataset.
```c++
 TarReader reader("test.tar");
 reader.read();
 for(const auto& archive: archives) {
    // Loads to CPU memory
    if(archive.second.device() == ArchiveDevice::CPU) {
          // Use buffer in CPU memory.
          archive.second.cpu_buffer();
    } else {
         // Use buffer in GPU memory.
         archive.second.gpu_buffer();
    }

 }
```
## Python(Pytorch) binding
TODO