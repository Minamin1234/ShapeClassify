import os
import numpy as np
import argparse
import glob
import tqdm
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def get_model(n, types=4):
    _input = layers.Input(shape=n)
    _l = layers.Dense(units=1024, activation="relu")(_input)
    _l = layers.Dense(units=2048, activation="relu")(_l)
    _l = layers.Dense(units=2048, activation="relu")(_l)
    _output = layers.Dense(units=types)

    _model = models.Model(inputs=_input, outputs=_output)
    _model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return _model

def train(_model: models.Model,
          _train_datas: np.ndarray,
          _train_labels: np.ndarray,
          _val_datas: np.ndarray,
          _val_labels: np.ndarray):
    _model.fit(
        _train_datas,
        _train_labels,
        epochs=10
    )
    _loss, _acc = _model.evaluate(_val_datas, _val_labels)
    print(f"loss: {_loss}, accuracy: {_acc}")
    return _model

def predict(_model: models.Model, _test_datas: np.ndarray) -> np.ndarray:
    _pred_model = tf.keras.Sequential([_model,
                                       layers.Softmax()])
    _pred = _pred_model.predict(_test_datas)
    return _pred

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--train_path", help="学習データのフォルダパス", default="train/train_data")
    argp.add_argument("--val_path", help="評価用データのフォルダパス", default="train/val_data")
    argp.add_argument("--test_path", help="評価用データのフォルダパス", default="train/test_data")
    argp.add_argument("--points", help="入力点数", default=4096)
    args = argp.parse_args()
    TRAIN = args.train_path
    VAL = args.val_path
    TEST = args.test_path
    POINTS = int(args.points)
    _train_files = glob.glob(os.path.join(TRAIN, "*.npz"))
    _val_files = glob.glob(os.path.join(VAL, "*.npz"))
    _test_files = glob.glob(os.path.join(TEST, "*.npz"))
    _train_datas = np.array([])
    _train_values = np.array([])
    _val_datas = np.array([])
    _val_values = np.array([])
    _test_datas = np.array([])
    _test_values = np.array([])
    for f in tqdm.tqdm(_train_files):
        _data = np.load(f, allow_pickle=True)
        _train_datas = np.append(_train_datas, [_data["pointcloud"]])
        _train_values = np.append(_train_values, _data["shape"])
        pass
    print(len(_train_datas))
    _train_values = np.reshape(_train_values, [2400, 4096, 3])
    print(_train_datas.shape)
    print(_train_values)
    #_model = get_model(POINTS, 4)
    pass