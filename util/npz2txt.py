import os
import numpy as np
import argparse
import tqdm
import glob

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("target_path")
    argp.add_argument("out_path")
    args = argp.parse_args()
    TARGET = args.target_path
    OUT = args.out_path
    _files = glob.glob(os.path.join(TARGET, "*.npz"))
    for _f in tqdm.tqdm(_files):
        _data = np.load(_f, allow_pickle=True)
        _fname = os.path.basename(_f).split(".")[0] + "_pcd.txt"
        np.savetxt(os.path.join(OUT, _fname),
                   _data["pointcloud"])
        pass
    pass