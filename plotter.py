import cupy as cp
import numpy as np
# import pyvista as pv
import matplotlib.pyplot as plt


class VelocityPlotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap, self.grid = colormap, grid
        # Build structured grid, nodal
        self.U, self.V = np.meshgrid(grid.u.arr.flatten(), grid.v.arr.flatten(), indexing='ij')

    def velocity_contourf(self, dist_slice, cb_lim=None, save=None):
        arr = np.real(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        if cb_lim is None:
            cb = np.linspace(np.amin(arr), np.amax(arr), num=100)
        else:
            cb = np.linspace(cb_lim[0], cb_lim[1], num=100)
        plt.figure()
        plt.contourf(self.U, self.V, arr, cb, extend='both')
        plt.xlabel('u'), plt.ylabel('v'), plt.colorbar()
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)

    def velocity_contourf_complex(self, dist_slice, title='Mode 0'):
        arr_r = np.real(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        arr_i = np.imag(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        arr_i[0, 0] += 1.0e-15

        cb_r = np.linspace(np.amin(arr_r), np.amax(arr_r), num=100)
        cb_i = np.linspace(np.amin(arr_i), np.amax(arr_i), num=100)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # cm = ax[0].contourf(self.U, self.V, arr_r, cb_r)
        cm = ax[0].pcolormesh(self.U, self.V, arr_r, shading='gouraud', vmin=cb_r[0], vmax=cb_r[-1], rasterized=True)
        fig.colorbar(cm, ax=ax[0])
        ax[0].set_xlabel('u'), ax[0].set_ylabel('v'), ax[0].set_title('Real')  # , ax[0].colorbar()
        # cm = ax[1].contourf(self.U, self.V, arr_i, cb_i)
        cm = ax[1].pcolormesh(self.U, self.V, arr_i, shading='gouraud', vmin=cb_i[0], vmax=cb_i[-1], rasterized=True)
        ax[1].set_xlabel('u'), ax[1].set_ylabel('v'), ax[1].set_title('Imag')  # , ax[1].colorbar()
        fig.colorbar(cm, ax=ax[1])

        plt.suptitle(title), plt.tight_layout()


class Plotter2D:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        self.X, self.Y = np.meshgrid(grid.x.arr.flatten(), grid.y.arr.flatten(), indexing='ij')
        self.KX, self.KY = np.meshgrid(grid.x.wavenumbers, grid.y.wavenumbers, indexing='ij')
        self.length_x, self.length_y = grid.x.length, grid.y.length

        # Finer grid (trig interpolation)
        self.grid_x_modes, self.grid_y_modes = grid.x.modes, grid.y.modes
        self.x_pad_width, self.y_pad_width = 50 * grid.x.pad_width, 50 * grid.y.pad_width
        self.x_fine = np.linspace(-grid.x.length/2, grid.x.length/2, num=self.grid_x_modes + 2 * self.x_pad_width,
                                  endpoint=False)
        self.y_fine = np.linspace(-grid.y.length/2, grid.y.length/2,
                                  num=self.trigonometric_interpolation(scalar=cp.asarray(self.X)).get().shape[1],
                                  endpoint=False)
        self.XF, self.YF = np.meshgrid(self.x_fine, self.y_fine, indexing='ij')

    def trigonometric_interpolation(self, scalar):
        scalar_spectrum = forward_scalar_transform(scalar)
        pad_spectrum = cp.zeros((self.grid_x_modes + 2 * self.x_pad_width,
                               self.grid_y_modes + self.y_pad_width)) + 0j
        pad_spectrum[self.x_pad_width:-self.x_pad_width, :-self.y_pad_width] = scalar_spectrum
        return inverse_scalar_transform(pad_spectrum)

    def scalar_plot_interpolated(self, scalar, title, cb_lim=None, save=None):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plot_arr = self.trigonometric_interpolation(scalar=scalar.arr_nodal).get()
        if cb_lim is None:
            cb = cp.linspace(cp.amin(plot_arr), cp.amax(plot_arr), num=100).get()
        else:
            cb = cp.linspace(cb_lim[0], cb_lim[1], num=100).get()
        plt.figure(figsize=(4, 10))
        plt.contourf(self.XF, self.YF, plot_arr, cb, extend='both', cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('y'), plt.title(title), plt.tight_layout()
        # plt.colorbar(), plt.tight_layout()

        if save is not None:
            plt.savefig(save)

    def spatial_scalar_plot(self, scalar, title, spectrum=True, save=None):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        cb = cp.linspace(cp.amin(scalar.arr_nodal), cp.amax(scalar.arr_nodal), num=100).get()

        plt.figure(figsize=(4, 10))
        plt.contourf(self.X, self.Y, scalar.arr_nodal.get(), cb, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('y'), plt.title(title), plt.tight_layout()
        # plt.colorbar()

        if spectrum:
            spectrum_abs = np.absolute(scalar.arr_spectral.get())

            cb_x = np.linspace(np.amin(spectrum_abs), np.amax(spectrum_abs), num=100)
            plt.figure()
            plt.contourf(self.KX, self.KY, spectrum_abs, cb_x)
            plt.colorbar(), plt.tight_layout()

        if save is not None:
            plt.savefig(save)

    def spatial_vector_plot(self, vector):
        if vector.arr_nodal is None:
            vector.inverse_fourier_transform()

        cb_x = cp.linspace(cp.amin(vector.arr_nodal[0, :, :]), cp.amax(vector.arr_nodal[0, :, :]), num=100).get()
        cb_z = cp.linspace(cp.amin(vector.arr_nodal[1, :, :]), cp.amax(vector.arr_nodal[1, :, :]), num=100).get()

        plt.figure()
        plt.contourf(self.X, self.Y, vector.arr_nodal[0, :, :].get(), cb_x, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.colorbar(), plt.tight_layout()

        plt.figure()
        plt.contourf(self.X, self.Y, vector.arr_nodal[1, :, :].get(), cb_z, cmap=self.colormap)
        plt.xlabel('x'), plt.ylabel('z')
        plt.colorbar(), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False, axis=False):
        time, series = time_in, series_in / (self.length_x * self.length_y)
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if axis:
            plt.axis([0, time[-1], 0, 1.1*np.amax(series)])
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            exact = 2 * 0.1 * 3.48694202e-01
            print('\nNumerical rate: {:0.10e}'.format(lin_fit[0]))
            # print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))  #
            print('cf. exact rate: {:0.10e}'.format(exact))
            print('The difference is {:0.10e}'.format(lin_fit[0] - exact))

    def show(self):
        plt.show()


def inverse_scalar_transform(scalar):
    return cp.fft.irfft2(cp.fft.fftshift(scalar, axes=0), norm='forward')


def forward_scalar_transform(scalar):
    return cp.fft.fftshift(cp.fft.rfft2(scalar, norm='forward'), axes=0)
