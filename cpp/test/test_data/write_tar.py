import io
import os
import tarfile

import numpy as np


def write_archive(tarsteam, size: int, name: str, value: int):
    array = np.full(size, value, dtype=np.uint8)
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.uname = "wei"
    tarstream.addfile(ti, io.BytesIO(array.tobytes()))

if __name__ == "__main__":
    tarstream = tarfile.open("test.tar", mode="w")
    write_archive(tarstream, 100, "foo_cpu", 2)
    write_archive(tarstream, 200, "boo_gpu", 4)
    write_archive(tarstream, 300, "poo_gpu", 6)
    tarstream.close()
