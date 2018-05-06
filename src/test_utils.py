import h5py
import numpy as np

DEFAULT_DATA_DIR = '../data/'

def make_test_data(data_dir=DEFAULT_DATA_DIR):
	f = h5py.File(data_dir+'test.h5', mode='w')
	f.create_dataset('img',data=np.random.rand(100,3,64,64))
	f.create_dataset('sound_embeddings',data=np.random.rand(100,128))
	f.create_dataset('class',data=np.random.randint(low=0,high=34,size=100))
	f.close()

make_test_data()
