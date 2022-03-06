import cupy as cp
import variables as var


class Static:
    """ Class for static fields governed by Gauss's law, here E_x """

    def __init__(self, resolutions):
        self.potential = var.SpaceScalar(resolutions=resolutions)
        self.electric = var.SpaceVector(resolutions=resolutions)

    def poisson(self, distribution, grid, invert=True):
        # Compute zeroth moment (in spectrum)
        distribution.compute_moment0(grid=grid)

        # Compute potential and field spectra
        self.potential.arr_spectral = grid.charge_sign * cp.nan_to_num(
            cp.divide(distribution.moment0.arr_spectral, grid.k_sq)
        )
        self.electric.arr_spectral[0, :, :] = -1j * cp.multiply(grid.x.device_wavenumbers[:, None],
                                                                self.potential.arr_spectral)
        self.electric.arr_spectral[1, :, :] = -1j * cp.multiply(grid.y.device_wavenumbers[None, :],
                                                                self.potential.arr_spectral)

        if invert:
            self.potential.inverse_fourier_transform()
            self.electric.inverse_fourier_transform()

    def compute_field_energy(self, grid):
        self.electric.inverse_fourier_transform()
        return self.electric.integrate_energy(grid=grid)
