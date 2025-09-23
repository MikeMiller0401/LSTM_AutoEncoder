#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:31:29 2020

@author: wenlong
"""

import numpy as np
import scipy.io as sio
import yaml
from easydict import EasyDict
from pathlib import Path
import h5py
import json
import codecs
import pathlib

def read_config(config_str):
    print("read_config")
    if config_str is None:
        print("no config")

    if type(config_str) is str or type(config_str) is pathlib.WindowsPath:
        config_path = Path(config_str)
        extension = config_path.suffix
        if extension == ".yaml":
            with open(config_path) as f:
                config = yaml.full_load(f)
                config = EasyDict(config)

        if extension == ".json":
            with codecs.open(config_path, "r", encoding="utf-8") as f:
                # 加载JSON数据
                config = json.load(f)
                config = EasyDict(config)

    return config


def read_data(data_str):
    print("read_data")
    if data_str is None:
        print("no data")

    if type(data_str) is str or type(data_str) is pathlib.WindowsPath:
        data_path = Path(data_str)
        extension = data_path.suffix
        if extension == ".mat":
            xy_data = sio.loadmat(data_str)
            return xy_data

        if extension == ".h5":
            dataV=h5py.File(data_str, 'r')

            xy_data = []
            for group in dataV.keys():
                print(group)
                # 根据一级组名获得其下面的组
                group_read = dataV[group]
                # 遍历该一级组下面的子组
                for subgroup in group_read.keys():
                    print(subgroup)
                    # 根据一级组和二级组名获取其下面的dataset
                    dset_read = dataV[group + '/' + subgroup]
                    # 遍历该子组下所有的dataset
                    dataset = dset_read[:]
                    print(dataset)
                    xy_data.append(dataset)

            # LabelV = np.zeros(dataV['Label'].shape)
            # ImagesV = np.zeros(dataV['TrainDataSet'].shape)
            # Label=dataV['Label']
            # Label.read_direct(LabelV)
            #
            # Images=dataV['TrainDataSet']
            # Images.read_direct(ImagesV)

            dataV.close()
            return xy_data

    if type(data_str) is np.ndarray:
        xy_data = data_str
        return xy_data

    return None


# def LoadMat(filename):
#     dataV=sio.loadmat(filename)
#     LabelV=dataV['Label']
#     ImagesV=dataV['Train']
#     return ImagesV, LabelV

#
# import OCSVM as o_svm

if __name__ == "__main__":

    # data_str = "D:/BaiduSyncdisk/AI_DataCleaning/AIMining/OCSVM/AD_data.mat"
    # xy_data = read_data(data_str)
    # print("xy_data")
    # # print(xy_data)
    #
    # data_str = "D:/BaiduSyncdisk/AI_DataCleaning/AIMining/OCSVM/example.h5"
    # xy_data = read_data(data_str)
    # print("xy_data")
    # # print(xy_data)
    #
    #
    # config_str = "D:/BaiduSyncdisk/AI_DataCleaning/AIMining/OCSVM/ocsvm.yaml"
    # config = read_config(config_str)
    # print("config")
    # print(config)


    config_str = "D:/BaiduSyncdisk/AI_DataCleaning/AIMining/OCSVM/labels.json"
    config = read_config(config_str)
    print("config")
    print(config)


    # o_svm.create_model(config)
    # print("config")


# def Save2MatV73(images, labels, filenameNew):
#     file = h5py.File(filenameNew, 'w')
#     file.create_dataset('TrainDataSet', data= images)
#     file.create_dataset('Label', data= labels)
#     file.close()
#
# def LoadMatV73(filename):
#     dataV=h5py.File(filename, 'r')
#     LabelV = np.zeros(dataV['Label'].shape)
#     ImagesV = np.zeros(dataV['TrainDataSet'].shape)
#     Label=dataV['Label']
#     Label.read_direct(LabelV)
#
#     Images=dataV['TrainDataSet']
#     Images.read_direct(ImagesV)
#
#     dataV.close()
#     return DataSet(ImagesV, LabelV)
#
# def LoadMatV73_T(filename):
#     print('Read Dataset: ', filename)
#     dataV=h5py.File(filename, 'r')
#     LabelV = np.zeros(dataV['Label'].shape)
#     ImagesV = np.zeros(dataV['TrainDataSet'].shape)
#
#     Label=dataV['Label']
#     Label.read_direct(LabelV)
#     Images=dataV['TrainDataSet']
#     Images.read_direct(ImagesV)
#
#     dataV.close()
#     print('Read Finished')
#     return DataSet(ImagesV.T, LabelV.T)
#
# def LoadMatV73_T_OH(filename):
#     print('Read Dataset: ', filename)
#     dataV=h5py.File(filename, 'r')
#     LabelV = np.zeros(dataV['Label'].shape)
#     ImagesV = np.zeros(dataV['TrainDataSet'].shape)
#
#     Label=dataV['Label']
#     Label.read_direct(LabelV)
#     Images=dataV['TrainDataSet']
#     Images.read_direct(ImagesV)
#
#     dataV.close()
#     print('Read Finished')
#     return DataSet(ImagesV.T, LabelV.T.reshape(-1))