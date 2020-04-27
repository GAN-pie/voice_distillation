#!/usr/bin/env python
# coding: utf-8

from os import path
import argparse
import time

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from data import IvectorSoftTargetSkyrim, IvectorSoftTargetMassEffect
from models import ModelStudentDual


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Teacher computes base knowledge from in-domain data.")
    parser.add_argument("data-lst", help="File containing data segment list.")
    parser.add_argument("test-lst", help="File containing development segments.")
    parser.add_argument("features", help="File containing feature vectors.")
    parser.add_argument("soft-path", help="File containing soft targets vectors.")
    parser.add_argument("output-dir", help="Give the final directory.")
    parser.add_argument("--input-dim", default=400, type=int, help="Define input dimension.")
    parser.add_argument("--batch", default=32, type=int, help="Set batch size.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Define temperature.")
    parser.add_argument("--lambda", default=0.5, type=float, help="Define imitation parameter.")
    parser.add_argument("--epochs", default=10, type=int, help="Set num of iteration.")
    parser.add_argument("--emb-dim", default=128, type=int, help="Set the embedding dimension.")
    parser.add_argument("--drop-rate", default=0.25, type=float, help="Set the dropout rate.")
    args = vars(parser.parse_args())

    lst_path = args["data-lst"]
    test_lst_path = args["test-lst"]
    features_path = args["features"]
    soft_target_path = args["soft-path"]
    out_dir = args["output-dir"]

    input_dim = args["input_dim"]
    epochs = args["epochs"]
    batch_size = args["batch"]
    drop_rate = args["drop_rate"]
    emb_dim = args["emb_dim"]
    temperature = args["temperature"]
    imitation = args["lambda"]

    start = time.time()

    # Dataset preparation
    corpus = path.splitext(path.basename(features_path))[0]
    if corpus == "skyrim":
        soft_data = IvectorSoftTargetSkyrim(lst_path, features_path, soft_target_path, dims=input_dim)
        soft_data_dev = IvectorSoftTargetSkyrim(test_lst_path, features_path, soft_target_path, dims=input_dim)
    else:
        soft_data = IvectorSoftTargetMassEffect(lst_path, features_path, soft_target_path, dims=input_dim)
        soft_data_dev = IvectorSoftTargetMassEffect(test_lst_path, features_path, soft_target_path, dims=input_dim)

    X_soft = soft_data.X
    Y_soft = soft_data.Y
    Y_hard = soft_data.Y_hard
    soft_data.save_scaler(path.join(out_dir, "data_scaler.pkl"))

    soft_data_dev.scaler = soft_data.scaler
    X_soft_dev = soft_data_dev.X
    Y_soft_dev = soft_data_dev.Y
    Y_hard_dev = soft_data_dev.Y_hard

    # Create model and train
    student = ModelStudentDual(soft_data.num_classes, soft_data.num_hard_classes, input_shape=(input_dim,), T=temperature,
                               imitation=imitation,  embedding_dim=emb_dim, drop_rate=drop_rate)

    ckpt = ModelCheckpoint(path.join(out_dir, "model_checkpoint.h5"), monitor="val_loss", save_best_only=True)
    # tb = TensorBoard(log_dir=path.join(out_dir, "graph"), histogram_freq=1, write_graph=True, write_images=True)
    # callb = [ckpt, tb]
    early = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10)
    callb = [ckpt, early]

    student.model.fit(x=X_soft, y=[Y_soft, Y_hard], batch_size=batch_size, epochs=epochs, shuffle=True,
                      callbacks=callb, validation_data=(X_soft_dev, [Y_soft_dev, Y_hard_dev]))
    student.save(out_dir)

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start))
