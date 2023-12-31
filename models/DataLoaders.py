import os
import torch
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import Dataset
from torch_geometric.data import Data
from models.utils import anidataloader
from torch_geometric.loader import DataLoader as PyGDataLoader

class ANI1Wrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_dir, seed):
        super(object, self).__init__()
        """
        batch_size: training batch size
        num_workers: Number of subprocesses to use for data loading, controls the parallelism
        valid_size: validation set percentage
        test_size: test set percentage
        data_dir: file path for the Data directory
        seed: random seed
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed

    def get_data_loaders(self):
        random_state = np.random.RandomState(seed=self.seed)
        # read all data with that ends wit hh5
        hdf5files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]

        n_mol = 0 # to record the total number of molecules/conformations
        self.train_species, self.valid_species, self.test_species = [], [], []
        self.train_positions, self.valid_positions, self.test_positions = [], [], []
        self.train_energies, self.valid_energies, self.test_energies = [], [], []
        self.test_smiles = []
        
        for f in hdf5files:
            print('reading:', f)
            h5_loader = anidataloader(os.path.join(self.data_dir, f))
            for data in h5_loader:
                # Get coordinates, atom types, energies, smiles
                X = data['coordinates']
                S = data['species']
                E = data['energies']
                # get the total number of conformations
                n_conf = E.shape[0]
                # create a list of indices and randomly shuffle them
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                # calculate the split points for training, validation, and test data
                split1 = int(np.floor(self.valid_size * n_conf))
                split2 = int(np.floor(self.test_size * n_conf))
                # split indices to 3
                valid_idx, test_idx, train_idx = \
                    indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

                # Record atom types, training energies, and positions for the 3 datasets
                self.train_species.extend([S] * len(train_idx))
                self.train_energies.append(E[train_idx])
                for i in train_idx:
                    self.train_positions.append(X[i])
                
                self.valid_species.extend([S] * len(valid_idx))
                self.valid_energies.append(E[valid_idx])
                for i in valid_idx:
                    self.valid_positions.append(X[i])
                
                self.test_species.extend([S] * len(test_idx))
                self.test_energies.append(E[test_idx])
                for i in test_idx:
                    self.test_positions.append(X[i])

                n_mol += 1
            
            h5_loader.cleanup()
        
        # Merge all lists of lists from all files into a single array
        self.train_energies = np.concatenate(self.train_energies, axis=0)
        self.valid_energies = np.concatenate(self.valid_energies, axis=0)
        self.test_energies = np.concatenate(self.test_energies, axis=0)

        print("# molecules:", n_mol)
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))
        print("# test conformations:", len(self.test_species))

        train_dataset = ANI1(
            self.data_dir, species=self.train_species,
            positions=self.train_positions, energies=self.train_energies
        )
        valid_dataset = ANI1(
            self.data_dir, species=self.valid_species,
            positions=self.valid_positions, energies=self.valid_energies
        )
        test_dataset = ANI1(
            self.data_dir, species=self.test_species, 
            positions=self.test_positions, energies=self.test_energies, 
            # smiles=self.test_smiles
        )

        """
        num_workers: specifies how many subprocesses to use for data loading. It controls the parallelism of data loading
        shuffle: determines whether the data should be shuffled at every epoch, turned on for training and off for others
        drop_last: controls whether the last batch should be dropped in case it is smaller than the specified batch_size
        pin_memory: when set to True, enables the DataLoader to use pinned memory for faster data transfer to CUDA-enabled GPUs
        persistent_workers: control whether the worker processes of the DataLoader should be kept alive across multiple iterations
        """
        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
        test_loader = PyGDataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=False
        )

        del self.train_species, self.valid_species, self.test_species
        del self.train_positions, self.valid_positions, self.test_positions
        del self.train_energies, self.valid_energies, self.test_energies

        return train_loader, valid_loader, test_loader
    

ATOM_DICT = {('Br', 0): 1, ('C', 0): 3, ('Cl', 0): 7, ('F', 0): 9, ('H', 0): 10, ('I', 0): 12, ('N', 0): 17,
            ('O', 0): 21, ('P', 0): 23, ('S', 0): 26}
SELF_INTER_ENERGY = {
    'H': -0.500607632585, 
    'C': -37.8302333826,
    'N': -54.5680045287,
    'O': -75.0362229210
}

class ANI1(Dataset):
    def __init__(self, data_dir, species, positions, energies):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies

    def __getitem__(self, index):
        # get the position, atoms, and energie for one conformation
        # Index automatically handlled by PyGDataLoader
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]

        x = []
        self_energy = 0.0

        for atom in atoms:
            # calculate cumulative self interaction energy
            x.append(ATOM_DICT[(atom, 0)])
            self_energy += SELF_INTER_ENERGY.get(atom, 0)

        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        # Hartree to kcal/mol
        y = torch.tensor(y, dtype=torch.float).view(1,-1) * 627.5
        # Hartree to kcal/mol
        self_energy = torch.tensor(self_energy, dtype=torch.float).view(1,-1) * 627.5
        data = Data(x=x, pos=pos, y=y-self_energy, self_energy=self_energy)

        return data

    def __len__(self):
        return len(self.positions)