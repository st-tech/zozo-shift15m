import argparse
import os
import shutil
import tarfile

from tqdm import tqdm


def _extract_tarfiles(data_dir):
    path = os.path.join(data_dir, "features")
    if not os.path.isdir(path):
        os.mkdir(path)

    image_tar_files = (
        open(os.path.join(data_dir, "tar_files.txt")).read().strip().split("\n")
    )
    image_tar_files = [os.path.join(data_dir, s) for s in image_tar_files]
    for fpath in tqdm(image_tar_files):
        with tarfile.open(fpath, "r") as tf:
            tf.extractall(data_dir)

        tmp_dir = fpath[:-7]
        for imgname in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, imgname)
            dst = os.path.join(data_dir, "features", imgname)
            shutil.move(src, dst)

        os.rmdir(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str)
    args = parser.parse_args()

    print("extracting cnn features")
    _extract_tarfiles(args.data_dir)
