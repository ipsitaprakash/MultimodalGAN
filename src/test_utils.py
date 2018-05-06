import h5py
import numpy as np

DEFAULT_DATA_DIR = '../data/'

def make_test_data(data_dir=DEFAULT_DATA_DIR):
	f = h5py.File(data_dir+'test.h5', mode='w')
	f.create_dataset('img',data=np.random.rand(100,64,64,3))
	f.create_dataset('embeddings',data=np.random.rand(100,128))
	f.create_dataset('class',data=np.ones(100))
	f.close()

make_test_data()