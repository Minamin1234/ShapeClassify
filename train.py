import os
import numpy as np
import argparse
import glob
import tqdm
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def get_model(n, types=4):
    _model = tf.keras.Sequential([
        layers.Flatten(input_shape=(n, 3)),
        layers.Dense(2048, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(types)
    ])

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
    print(_train_datas.shape)
    print(_train_labels.shape)
    _model.fit(
        _train_datas,
        _train_labels,
        epochs=10,
        batch_size=32,
        use_multiprocessing=True
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
    _train_file = glob.glob(os.path.join(TRAIN, "*.npz"))[0]
    _val_file = glob.glob(os.path.join(VAL, "*.npz"))[0]
    _test_file = glob.glob(os.path.join(TEST, "*.npz"))[0]
    _train_datas = np.array([])
    _train_values = np.array([])
    _val_datas = np.array([])
    _val_values = np.array([])
    _test_datas = np.array([])
    _test_values = np.array([])

    _train_data = np.load(_train_file, allow_pickle=True)
    _train_datas = _train_data["pointcloud"]
    _train_values = _train_data["shape"]
    
    _val_data = np.load(_test_file, allow_pickle=True)
    _val_datas = _val_data["pointcloud"]
    _val_values = _val_data["shape"]

    _test_data = np.load(_test_file, allow_pickle=True)
    _test_datas = _test_data["pointcloud"]
    _test_values = _test_data["shape"]
    
    _model = get_model(POINTS, 4)
    _model = train(_model, _train_datas, _train_values, _val_datas, _val_values)
    _idxs = np.arange(4)
    rng = np.random.default_rng()
    rng.shuffle(_idxs)
    MODEL_NAMES = np.array([
        "rect",
        "triangle",
        "circle",
        "star"
    ])
    _test_datas = _test_datas[_idxs]
    _test_values = _test_values[_idxs]
    _pred = predict(_model, _test_datas)
    for i in range(len(_pred)):
        print(MODEL_NAMES[np.argmax(_pred[i])])
    print(MODEL_NAMES[_test_values.astype(int)])
    pass