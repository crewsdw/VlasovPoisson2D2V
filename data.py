import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self, distribution, density, potential):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1],
                                       distribution.shape[2], distribution.shape[3],
                                       distribution.shape[4], distribution.shape[5]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density]),
                             chunks=True,
                             maxshape=(None, density.shape[0], density.shape[1]),
                             dtype='f')
            f.create_dataset('potential', data=np.array([potential]),
                             chunks=True,
                             maxshape=(None, potential.shape[0], potential.shape[1]),
                             dtype='f')
            f.create_dataset('time', data=[0.0], chunks=True, maxshape=(None,))
            f.create_dataset('total_energy', data=[], chunks=True, maxshape=(None,))
            f.create_dataset('total_density', data=[], chunks=True, maxshape=(None,))

    def save_data(self, distribution, density, potential, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new timeline
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['potential'].resize((f['potential'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['density'][-1] = density
            f['potential'][-1] = potential
            f['time'][-1] = time

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            pdf = f['pdf'][()]
            den = f['density'][()]
            eng = f['potential'][()]
            total_eng = f['total_energy'][()]
            total_den = f['total_density'][()]
        return time, pdf, den, eng, total_eng, total_den
