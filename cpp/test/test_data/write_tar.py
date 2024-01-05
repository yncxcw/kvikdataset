import io
import os
import tarfile

tarstream = tarfile.open("test.tar", mode="w")

for i in range(1, 4):
    size = 100 * i
    random_bytes=io.BytesIO(os.urandom(size))
    ti = tarfile.TarInfo(f"field-{i}")
    ti.size = size
    ti.uname = "wei"
    tarstream.addfile(ti, random_bytes)

tarstream.close()
