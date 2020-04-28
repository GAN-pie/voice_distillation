#!/usr/bin/env python
# coding: utf-8

from os import path
import argparse
import time

from keras.models import load_model, Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from data import IvectorSkyrim, IvectorMassEffect, IvectorVoxceleb
from models import custom_softmax


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract embedding from given model.")
    parser.add_argument("data-lst", help="File containing data segment list.")
    parser.add_argument("features", help="File containing feature vectors.")
    parser.add_argument("model-dir", help="Give the model directory.")
    parser.add_argument("--input-dim", default=400, type=int, help="Define input dimension.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Define temperature")
    args = vars(parser.parse_args())

    lst_path = args["data-lst"]
    features_path = args["features"]
    mdl_dir = args["model-dir"]

    input_dim = args["input_dim"]
    temperature = args["temperature"]

    start = time.time()

    corpus = path.splitext(path.basename(features_path))[0]
    scaler_path = path.join(mdl_dir, "data_scaler.pkl")
    if "masseffect" in corpus:
        data = IvectorMassEffect(lst_path, features_path, dims=input_dim, scaler=scaler_path)
    elif "skyrim" in corpus:
        data = IvectorSkyrim(lst_path, features_path, dims=input_dim, scaler=scaler_path)
    elif "voxceleb" in corpus:
        data = IvectorVoxceleb(lst_path, features_path, dims=input_dim, scaler=scaler_path)
    X = data.X
    Y = data.Y

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    set_session(sess)

    custom_obj = {"activation_fn": custom_softmax(temperature)}
    model = load_model(path.join(mdl_dir, "model_checkpoint.h5"), custom_objects=custom_obj)

    embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
    embeddings = embedding_layer_model.predict(X)

    location = path.join(mdl_dir, "_".join([
        path.splitext(path.basename(features_path))[0],
        "pvectors.txt"]))
    with open(location, "w") as fd:
        lines = []
        for i, k in enumerate(data.ivectors.keys()):
            array_str = " ".join(list(map(str, embeddings[i].tolist())))
            lines += ["{} {}".format(k, array_str)]
        fd.write("\n".join(lines))
    print("p-vectors saved to: {}".format(location))

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start))
