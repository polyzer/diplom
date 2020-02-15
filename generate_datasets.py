import h5py, os
import caffe
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import glob
import pandas as pd
import argparse

IMG_SIZE = 200


def generate_h5_dataset(train_files):

    df = pd.read_csv(train_files)
    print(df.head())
    print(df.shape)
    #exit()
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    X = np.zeros( (df.shape[0], IMG_SIZE, IMG_SIZE), dtype='f4' ) 
    y = np.zeros( (df.shape[0], 1), dtype='f4' )
    for index, row in df.iterrows():
        label = row["label"]
        f_name = row["filenames"]
        img = np.load(f_name)
        y[index] = int(label)
        X[index] = img
    with h5py.File('train.h5','w') as H:
        H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
        H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
    with open('train_h5_list.txt','w') as L:
        L.write( 'train.h5' ) # list all h5 files you are going to use


def generate_labeling_textfile(in_dir, save_file_name):
    # f_filenames = glob.glob(os.path.join(in_dir, "F*.npy"))
    # m_filenames = glob.glob(os.path.join(in_dir, "M*.npy"))
    f_names = glob.glob(os.path.join(in_dir, "*.npy"))
    save_file = open(os.path.join(save_file_name), "w")
    counter = 0
    df = pd.DataFrame({
        "filenames": f_names
    })
    df.to_csv(save_file_name, index_label="label")
    # for idx, f_name in enumerate(f_filenames):
    #     file_name = 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataset generation file')
    parser.add_argument('--type', type=int, help='File name to save data')
    parser.add_argument('--npy_dir', type=str,  help='NPY files directory')
    parser.add_argument('--save_name', type=str, help='File name to save data')
    parser.add_argument('--train_name', type=str, help='File to load from')

    args = parser.parse_args()

    if args.type == 1:
        generate_labeling_textfile(args.npy_dir, args.save_name)
    if args.type == 2:
        generate_h5_dataset(args.train_name)
