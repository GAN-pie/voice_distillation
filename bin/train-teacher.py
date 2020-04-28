#!/usr/bin/env python
# coding: utf-8

from os import path
import argparse
import time

import numpy as np

from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Model

from data import IvectorMassEffect, IvectorSkyrim
from models import ModelTeacher


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Teacher computes base knowledge from in-domain data.")
    parser.add_argument("data-lst", help="File containing data segment list.")
    parser.add_argument("test-lst", help="File containing development segments.")
    parser.add_argument("features", help="File containing feature vectors.")
    parser.add_argument("output-dir", help="Give the final directory.")
    parser.add_argument("--soft-lst", help="Files containing data to be softed with trained model.")
    parser.add_argument("--soft-features", help="Files containing features to be softed with trained model.")
    parser.add_argument("--input-dim", default=400, type=int, help="Define input dimension.")
    parser.add_argument("--batch", default=32, type=int, help="Set batch size.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Define temperature")
    parser.add_argument("--epochs", default=10, type=int, help="Set num of iteration.")
    parser.add_argument("--emb-dim", default=128, type=int, help="Set the embedding dimension.")
    parser.add_argument("--drop-rate", default=0.25, type=float, help="Set the dropout rate.")
    args = vars(parser.parse_args())

    lst_path = args["data-lst"]
    test_lst_path = args["test-lst"]
    features_path = args["features"]
    out_dir = args["output-dir"]

    input_dim = args["input_dim"]
    soft_lst_path = args["soft_lst"]
    soft_features_path = args["soft_features"]
    epochs = args["epochs"]
    batch_size = args["batch"]
    drop_rate = args["drop_rate"]
    emb_dim = args["emb_dim"]
    temperature = args["temperature"]

    start = time.time()

    # Prepare dataset
    corpus = path.splitext(path.basename(features_path))[0]
    if "masseffect" in corpus:
        train = IvectorMassEffect(lst_path, features_path, dims=input_dim)
        dev = IvectorMassEffect(test_lst_path, features_path, dims=input_dim)
    else:
        train = IvectorSkyrim(lst_path, features_path, dims=input_dim)
        dev = IvectorSkyrim(test_lst_path, features_path, dims=input_dim)
    X = train.X
    Y = train.Y
    train.save_scaler(path.join(out_dir, "data_scaler.pkl"))

    dev.scaler = train.scaler
    X_dev = dev.X
    Y_dev = dev.Y

    # Model training
    teacher = ModelTeacher(train.num_classes, input_shape=(input_dim,), T=temperature, embedding_dim=emb_dim, drop_rate=drop_rate)

    ckpt = ModelCheckpoint(path.join(out_dir, "model_checkpoint.h5"), monitor="val_loss", save_best_only=True)
    tb = TensorBoard(log_dir=path.join(out_dir, "graph"), histogram_freq=1, write_graph=True, write_images=True)
    early = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=4)

    callb = [ckpt, early]

    teacher.model.fit(X, Y, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=callb, validation_data=(X_dev, Y_dev))
    teacher.save(out_dir)

    # Model evaluation on validation set
    dev_predictions = teacher.model.predict(X_dev)
    dev_classes = np.argmax(dev_predictions, axis=1).tolist()

    fscores = f1_score(dev.labels, dev_classes, average=None)
    fscores_str = "\n".join(map(lambda i: "{0:25s}: {1:.4f}".format(
        dev._le.classes_[i], fscores[i]), range(len(fscores))))
    print("F1-scores for each classes:\n{}".format(fscores_str))
    accuracy = accuracy_score(dev.labels, dev_classes)
    print("Accuracy: {0:.4f}".format(accuracy))
    with open(path.join(out_dir, "eval.log"), "w") as fd:
        print("F1-scores for each classes:\n{}".format(fscores_str), file=fd)
        print("Accuracy: {0:.4f}".format(accuracy), file=fd)

    confusions = confusion_matrix(dev.labels, dev_classes)

    plt.figure()
    plt.title("Confusion matrix")
    seaborn.heatmap(confusions, xticklabels=dev._le.classes_, yticklabels=dev._le.classes_,
                    linewidths=0.6, fmt="d", annot=True, cmap="YlGnBu")
    plt.tight_layout()
    loc = path.join(out_dir, path.splitext(path.basename(test_lst_path))[0]+"_confusion_matrix.pdf")
    plt.savefig(loc)
    plt.close()
    print("confusion matrix saved to: {}".format(loc))

    embedding_layer_model = Model(inputs=teacher.model.input, outputs=teacher.model.get_layer("embedding").output)
    embeddings = embedding_layer_model.predict(X_dev)
    estimator = TSNE(perplexity=20.0)
    embeddings_prime = estimator.fit_transform(embeddings)
    fig, axe = plt.subplots(1, 1)
    axe.scatter(embeddings_prime[:, 0], embeddings_prime[:, 1], c=dev.labels)
    plt.savefig(path.join(out_dir, "_".join([
        path.splitext(path.basename(test_lst_path))[0],
        path.splitext(path.basename(features_path))[0],
        "embeddings.pdf"]))
    )

    # Predict soft targets on given data if any
    if soft_lst_path and soft_features_path:
        soft_corpus = path.splitext(path.basename(soft_features_path))[0]
        if soft_corpus == "skyrim":
            soft_data = IvectorSkyrim(soft_lst_path, soft_features_path, dims=input_dim)
        else:
            soft_data = IvectorMassEffect(soft_lst_path, soft_features_path, dims=input_dim)

        soft_data.scaler = train.scaler
        X_soft = soft_data.X

        print("predicting softtargets...")
        softtargets = teacher.model.predict(X_soft)

        location = path.join(out_dir, "_".join([
            path.splitext(path.basename(soft_features_path))[0],
            "soft_targets.txt"]))
        with open(location, "w") as fd:
            lines = []
            for i, k in enumerate(soft_data.ivectors.keys()):
                array_str = " ".join(list(map(str, softtargets[i].tolist())))
                lines += ["{} {}".format(k, array_str)]
            fd.write("\n".join(lines))
        print("soft targets saved to: {}".format(location))

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start))
