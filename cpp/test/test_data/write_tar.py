import io
import os
import tarfile


def write_archive(tarsteam, size: int, name: str):
    random_bytes=io.BytesIO(os.urandom(size))
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.uname = "wei"
    tarstream.addfile(ti, random_bytes)

if __name__ == "__main__":
    tarstream = tarfile.open("test.tar", mode="w")
    write_archive(tarstream, 16, "foo_cpu")
    write_archive(tarstream, 200, "boo_gpu")
    write_archive(tarstream, 300, "poo_gpu")
    tarstream.close()
