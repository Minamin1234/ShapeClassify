import os
import numpy as np
import argparse
from stl import mesh
import math

def generate_mesh(_vertices: np.ndarray, _faces: np.ndarray):
    _mesh = mesh.Mesh(np.zeros(_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(_faces):
        for j in range(3):
            _mesh.vectors[i][j] = _vertices[f[j],:]
            pass
        pass
    return _mesh

def save_stl(_mesh: mesh.Mesh, path: str) -> None:
    _mesh.save(f"{path}")
    pass

def rect(height: float, width: float) -> [np.ndarray, np.ndarray]:
    _vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, height, 0.0],
        [width, height, 0.0],
        [width, 0.0, 0.0]
    ])
    _faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    return _vertices, _faces

def triangle(height: float, width: float) -> [np.ndarray, np.ndarray]:
    _vertices = np.array([
        [0.0, 0.0, 0.0],
        [width / 2, height, 0.0],
        [width, 0.0, 0.0]
    ])
    _faces = np.array([
        [0, 1, 2]
    ])

    return _vertices, _faces

def circle(radius: float, deg: float=5) -> [np.ndarray, np.ndarray]:
    _vertices = np.array([
        [0.0, 0.0, 0.0]
    ])
    _faces = np.array([])
    for _deg in range(0, 360, deg):
        _v = [radius * math.cos(math.radians(_deg)),
              radius * math.sin(math.radians(_deg)),
              0.0]
        _vertices = np.append(_vertices, _v)
        pass
    _vertices = np.reshape(_vertices, [int(len(_vertices) / 3), 3])
    for i in range(1, int(360/deg)+1):
        _f = [i, 0, (i + 1) % int(len(_vertices))]
        if _f[2] == 0:
            _f[2] = 1
            pass
        _faces = np.append(_faces, _f)
        pass
    _faces = np.reshape(_faces, [int(len(_faces) / 3), 3]).astype(int)
    return _vertices, _faces

def star(radius: float) -> [np.ndarray, np.ndarray]:
    _vertices = np.array([
        [0.0, 0.0, 0.0]
    ])
    _faces = np.array([])
    _div = int(360 / 5)
    for i in range(5):
        _deg = (360/5) * i
        _v = [radius * math.cos(math.radians(_deg)),
              radius * math.sin(math.radians(_deg)),
              0.0]
        _vertices = np.append(_vertices, _v)
        pass
    _faces = np.array([
        [1, 0, 3],
        [1, 0, 4],
        [2, 0, 4],
        [2, 0, 5],
        [3, 0, 5],
        [3, 0, 1],
        [4, 0, 1],
        [4, 0, 2],
        [5, 0, 2],
        [5, 0, 3]
    ])
    _vertices = np.reshape(_vertices, [int(len(_vertices) / 3), 3])
    return _vertices, _faces