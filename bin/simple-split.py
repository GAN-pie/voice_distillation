#!/usr/bin/env python3
# coding: utf-8

import os
from os import path
import sys
import random
import math
import copy
import argparse


def read_lst_file(fname):
    with open(fname, "r") as fd:
        return [line.strip() for line in fd]


def write_lst(fname, lst):
    with open(fname, "w") as fd:
        for x in lst:
            print(x, file=fd)


class Spliter:
    """The Spliter class preserves a soft equivalence between english and
    french list."""

    def __init__(self, english, french, separator, minimum):
        self._separator = separator
        self._minimum = minimum

        self._en_lst = read_lst_file(english)
        self._fr_lst = read_lst_file(french)

        assert len(self._en_lst) == len(self._fr_lst)
        self._filtered = False

        self._filter()

    def get_labels(self):
        return list(map(lambda x: x.split(self._separator)[1], self._en_lst))

    def get_unique_labels(self):
        return list(set(self.get_labels()))

    def _filter(self):
        tmp_en = {}
        for e in self._en_lst:
            label = e.split(self._separator)[1]
            if tmp_en.get(label):
                tmp_en[label] += [e]
            else:
                tmp_en[label] = [e]

        tmp_fr = {}
        for e in self._fr_lst:
            label = e.split(self._separator)[1]
            if tmp_fr.get(label):
                tmp_fr[label] += [e]
            else:
                tmp_fr[label] = [e]

        shared = set(tmp_en.keys()).intersection(set(tmp_fr.keys()))

        # for e in list(tmp_en.keys()):
        #     if e not in shared:
        #         del tmp_en[e]
        # for f in list(tmp_fr.keys()):
        #     if f not in shared:
        #         del tmp_fr[f]

        new_en_lst = []
        new_fr_lst = []
        for l in shared:
            if (len(tmp_en[l]) >= self._minimum) and (len(tmp_fr[l]) >= self._minimum):
                # copy_list = copy.deepcopy(tmp_en[l])
                random.shuffle(tmp_en[l])
                random.shuffle(tmp_fr[l])
                new_en_lst += copy.deepcopy(tmp_en.get(l)[0:self._minimum])
                new_fr_lst += copy.deepcopy(tmp_fr.get(l)[0:self._minimum])

        # for l in shared:
        #     if len(tmp_fr[l]) >= self._minimum:
        #         copy_list = copy.deepcopy(tmp_fr[l])
        #         random.shuffle(copy_list)
        #         new_fr_lst += copy_list[:self._minimum]

        # assert tmp_en.keys() == tmp_fr.keys()

        self._en_lst = new_en_lst
        self._fr_lst = new_fr_lst

        assert len(self._en_lst) == len(self._fr_lst)
        self._filtered = True

    def split(self):
        if not self._filtered:
            self._filter()

        tmp = {}
        for i in range(len(self._en_lst)):
            label = self._en_lst[i].split(self._separator)[1]
            if tmp.get(label):
                tmp[label] += [(self._en_lst[i], self._fr_lst[i])]
            else:
                tmp[label] = [(self._en_lst[i], self._fr_lst[i])]

        train = []
        test = []
        delim = math.floor(self._minimum * 0.8)
        for label in tmp.keys():
            train += tmp[label][:delim]
            test += tmp[label][delim:]

        test_en_lst, test_fr_lst = list(zip(*test))
        train_en_lst, train_fr_lst = list(zip(*train))

        # test_en_label = list(map(lambda x: x.split(self._separator)[1], test_en_lst))
        # test_fr_label = list(map(lambda x: x.split(self._separator)[1], test_fr_lst))
        # train_en_label = list(map(lambda x: x.split(self._separator)[1], train_en_lst))
        # train_fr_label = list(map(lambda x: x.split(self._separator)[1], train_fr_lst))
        #
        # try:
        #     assert set(test_en_label) == set(test_fr_label)
        # except AssertionError:
        #     print(list(zip(set(test_en_label), set(test_fr_label))))
        #     exit()
        #
        # assert set(train_en_label) == set(train_fr_label)

        return (train_en_lst, train_fr_lst, test_en_lst, test_fr_lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("english")
    parser.add_argument("french")
    parser.add_argument("--minimum", default=160, type=int)
    parser.add_argument("--field-separator", default=",", choices=[",", "."])
    parser.add_argument("--output-dir", default="./")

    options = vars(parser.parse_args())

    sep = options["field_separator"]
    minimum = options["minimum"]
    out_dir = options["output_dir"]

    en_lst_path = options["english"]
    fr_lst_path = options["french"]

    spliter = Spliter(en_lst_path, fr_lst_path, sep, minimum)

    train_en, train_fr, test_en, test_fr = spliter.split()
    write_lst(path.join(out_dir, "train_en.lst"), train_en)
    write_lst(path.join(out_dir, "train_fr.lst"), train_fr)
    write_lst(path.join(out_dir, "val_en.lst"), test_en)
    write_lst(path.join(out_dir, "val_fr.lst"), test_fr)
