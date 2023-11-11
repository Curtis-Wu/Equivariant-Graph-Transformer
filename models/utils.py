# Written by Roman Zubatyuk and Justin S. Smith
import h5py
import numpy as np
import os
import math


class anidataloader(object):

    ''' Contructor '''
    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - '+store_file)
        self.store = h5py.File(store_file)

    ''' Group recursive iterator (iterate through all groups in all branches and return datasets in dicts) '''
    def h5py_dataset_iterator(self,g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset): # test for dataset
                data = {'path':path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k])

                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii') for a in dataset]

                        data.update({k:dataset})

                yield data
            else: # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        for data in self.h5py_dataset_iterator(g):
            yield data

    ''' Returns the requested dataset '''
    def get_data(self, path, prefix=''):
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        # print(path)
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k].value)

                if type(dataset) is np.ndarray:
                    if dataset.size != 0:
                        if type(dataset[0]) is np.bytes_:
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    ''' Returns the number of groups '''
    def group_size(self):
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()


def ani1x_iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for smi, grp in f.items():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=np.bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            d['smi'] = smi
            yield d 


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = (config['lr'] - config['min_lr']) * epoch / config['warmup_epochs'] + config['min_lr']
    elif epoch < config['warmup_epochs'] + config['patience_epochs']:
        lr = config['lr']
    else:
        prev_epochs = config['warmup_epochs'] + config['patience_epochs']
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - prev_epochs) / (config['epochs'] - prev_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr