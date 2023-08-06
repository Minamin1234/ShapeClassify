import os
import numpy as np
import argparse
from stl import mesh
import math
import random
import tqdm
import open3d as o3d
import model.model_2d

def get_random_params():
    _height = random.uniform(0.5, 12.0)
    _width = random.uniform(0.5, 12.0)
    _radius = random.uniform(0.1, 10.0)
    return _height, _width, _radius

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("out_path")
    argp.add_argument("--models", default=1)
    argp.add_argument("--points", default=4096)
    args = argp.parse_args()
    OUT = args.out_path
    MODELS = int(args.models)
    POINTS = int(args.points)
    MODEL_NAMES = [
        "rect",
        "triangle",
        "circle",
        "star"
    ]
    _datas = np.array([])
    _values = np.array([])
    for i in tqdm.tqdm(range(MODELS)):
        for t in range(4):
            _h, _w, _r = get_random_params()
            if t == 0:
                _v, _f = model.model_2d.rect(_h, _w)
                pass
            elif t == 1:
                _v, _f = model.model_2d.triangle(_h, _w)
                pass
            elif t == 2:
                _v, _f = model.model_2d.circle(_r)
                pass
            elif t == 3:
                _v, _f = model.model_2d.star(_r)
                pass
            _m = model.model_2d.generate_mesh(_v, _f)
            _stl_folder = os.path.join(OUT, "stl")
            try:
                os.mkdir(_stl_folder)
            except:
                pass
            _stl_name = os.path.join(_stl_folder, f"{MODEL_NAMES[t]}_{i+1}.stl")
            model.model_2d.save_stl(_m,
                                    _stl_name)
            _stl = o3d.io.read_triangle_mesh(_stl_name)
            _pcd = np.array(_stl.sample_points_poisson_disk(number_of_points=POINTS).points)
            _datas = np.append(_datas, np.array(_pcd))
            _values = np.append(_values, t)
            """np.savez_compressed(os.path.join(OUT, f"{MODEL_NAMES[t]}_{i+1}.npz"),
                                pointcloud=_pcd,
                                shape=t)"""
            _txt_folder = os.path.join(OUT, "txt")
            _txt_name = os.path.join(_txt_folder, f"{MODEL_NAMES[t]}_{i+1}_pcd.txt")
            try:
                os.mkdir(_txt_folder)
            except:
                pass
            np.savetxt(_txt_name,
                       _pcd)
            pass
        pass
    _datas = np.reshape(_datas, [MODELS * 4, POINTS, 3])
    np.savez_compressed(
        os.path.join(OUT, f"data.npz"),
        pointcloud=_datas,
        shape=_values
    )
    pass

