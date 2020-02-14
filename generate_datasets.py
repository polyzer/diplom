import h5py, os
import caffe
import numpy as np
import lmdb
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import glob
import pandas as pd
import argparse

IMG_SIZE = 224

def write_images_to_lmdb(img_dir, db_name):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = IMG_SIZE*IMG_SIZE*3*2*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(files):
            X = mp.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])

def read_images_from_lmdb(db_name, visualize):
	env = lmdb.open(db_name)
	txn = env.begin()
	cursor = txn.cursor()
	X = []
	y = []
	idxs = []
	for idx, (key, value) in enumerate(cursor):
		datum = caffe_pb2.Datum()
		datum.ParseFromString(value)
		X.append(np.array(datum_to_array(datum)))
		y.append(datum.label)
		idxs.append(idx)
	if visualize:
	    print "Visualizing a few images..."
	    for i in range(9):
	        img = X[i]
	        plt.subplot(3,3,i+1)
	        plt.imshow(img)
	        plt.title(y[i])
	        plt.axis('off')
	    plt.show()
	print " ".join(["Reading from", db_name, "done!"])
	return X, y, idxs

def generate_h5_dataset(imgs_folder):
    SIZE = 224 # fixed size to all images
    with open( 'train.txt', 'r' ) as T :
        lines = T.readlines()
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    X = np.zeros( (len(lines), SIZE, SIZE, 3), dtype='f4' ) 
    y = np.zeros( (len(lines), 1), dtype='f4' )
    for i,l in enumerate(lines):
        sp = l.split(' ')
        img = np.load('/tmp/123.npy')
        #img = caffe.io.load_image( sp[0] )
        #img = caffe.io.resize( img, (SIZE, SIZE, 3) ) # resize to fixed size
        # you may apply other input transformations here...
        # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
        # for example
        # transposed_img = img.transpose((2,0,1))[::-1,:,:] # RGB->BGR
        #X[i] = transposed_img
        y[i] = float(sp[1])
    with h5py.File('train.h5','w') as H:
        H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
        H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
    with open('train_h5_list.txt','w') as L:
        L.write( 'train.h5' ) # list all h5 files you are going to use


def generate_labeling_textfile(in_dir, save_dir, save_file_name):
    # f_filenames = glob.glob(os.path.join(in_dir, "F*.npy"))
    # m_filenames = glob.glob(os.path.join(in_dir, "M*.npy"))
    f_names = glob.glob(os.path.join(in_dir, "*.npy"))
    save_file = open(os.path.join(save_dir, save_file_name), "w")
    counter = 0
    df = pd.DataFrame({
        "filenames": f_names
    })
    df.to_csv(save_file_name)
    # for idx, f_name in enumerate(f_filenames):
    #     file_name = 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataset generation file')
    parser.add_argument('store_path',  help='NPY files directory')
    parser.add_argument('save_name',  help='File name to save data')
    
    
