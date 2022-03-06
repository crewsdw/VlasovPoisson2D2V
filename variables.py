import cupy as cp
import numpy as np
import dispersion
import scipy.optimize as opt


class SpaceScalar:
    """ Class for configuration-space scalars """

    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), norm='forward')

    def integrate(self, grid, array):
        """ Integrate an array, possibly self """
        # arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        arr_add = cp.zeros((self.res_x + 1, self.res_y + 1))
        arr_add[:-1, :-1] = array
        arr_add[-1, :-1] = array[0, :]
        arr_add[:-1, -1] = array[:, 0]
        arr_add[-1, -1] = array[0, 0]
        return trapz(arr_add, grid.x.dx, grid.y.dx)

    def integrate_energy(self, grid):
        self.integrate(grid=grid, array=0.5 * self.arr_nodal ** 2.0)


class SpaceVector:
    """ Class for configuration-space vectors """

    def __init__(self, resolutions):
        self.resolutions = resolutions
        self.arr_nodal, self.arr_spectral = None, None
        self.arr_nodal = cp.zeros((2, resolutions[0], resolutions[1]))
        self.init_spectral_array()

        self.energy = SpaceScalar(resolutions=resolutions)

    def init_spectral_array(self):
        if self.arr_spectral is not None:
            return
        else:
            x_spec = cp.fft.rfft2(self.arr_nodal[0, :, :])
            y_spec = cp.fft.rfft2(self.arr_nodal[1, :, :])
            self.arr_spectral = cp.array([x_spec, y_spec])

    def integrate_energy(self, grid):
        self.inverse_fourier_transform()
        self.energy.arr_nodal = self.arr_nodal[0, :, :] ** 2.0 + self.arr_nodal[1, :, :] ** 2.0
        return 0.5 * self.energy.integrate(grid=grid, array=self.energy.arr_nodal)

    def fourier_transform(self):
        self.arr_spectral[0, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[0, :, :], norm='forward'), axes=0)
        self.arr_spectral[1, :, :] = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal[1, :, :], norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal[0, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[0, :, :], axes=0), norm='forward')
        self.arr_nodal[1, :, :] = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral[1, :, :], axes=0), norm='forward')


class VelocityScalar:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order
        # arrays
        self.arr_nodal = None


class Distribution:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order

        # arrays
        self.arr_spectral, self.arr_nodal = None, None
        self.moment0, self.moment2 = (SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]]),
                                      SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]]))
        self.moment1 = SpaceVector(resolutions=[self.resolutions[0], self.resolutions[1]])
        self.moment1_magnitude = SpaceScalar(resolutions=[self.resolutions[0], self.resolutions[1]])

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, axes=(0, 1), norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), axes=(0, 1), norm='forward')

    def total_density(self, grid):
        self.compute_moment0(grid=grid)
        return self.moment0.integrate(grid=grid, array=self.moment0.arr_nodal)

    def total_thermal_energy(self, grid):
        self.compute_moment2(grid=grid)
        return 0.5 * self.moment2.integrate(grid=grid, array=self.moment2.arr_nodal)

    def compute_moment0(self, grid):
        self.moment0.arr_spectral = grid.moment0(variable=self.arr_spectral)
        self.moment0.inverse_fourier_transform()

    def compute_moment1(self, grid):
        self.moment1.arr_spectral[0, :, :] = grid.moment1_u(variable=self.arr_spectral)
        self.moment1.arr_spectral[1, :, :] = grid.moment1_v(variable=self.arr_spectral)
        self.moment1.inverse_fourier_transform()

    def compute_moment1_magnitude(self):
        self.moment1_magnitude.arr_nodal = cp.sqrt(cp.square(self.moment1.arr_nodal[0, :, :]) +
                                                   cp.square(self.moment1.arr_nodal[1, :, :]))

    def compute_moment2(self, grid):
        self.moment2.arr_spectral = grid.moment2(variable=self.arr_spectral)
        self.moment2.inverse_fourier_transform()

    def set_modal_distribution(self, idx, velocity_scalar):
        velocity_scalar.arr_nodal = self.arr_spectral[idx[0], idx[1], :, :, :, :]

    def nodal_flatten(self):
        return self.arr_nodal.reshape(self.resolutions[0], self.resolutions[1],
                                      self.resolutions[2] * self.order, self.resolutions[3] * self.order)

    def initialize_two_stream(self, grid, vb):
        # space indicators
        ix, iy = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.y.device_arr)
        x, y = cp.tensordot(grid.x.device_arr, iy, axes=0), cp.tensordot(ix, grid.y.device_arr, axes=0)

        # velocity indicators
        iu, iv = cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        u, v = cp.tensordot(grid.u.device_arr, iv, axes=0), cp.tensordot(iu, grid.v.device_arr, axes=0)

        # set basic-ass maxwellian
        factor = 1 / (2 * cp.pi)
        maxwell1 = factor * cp.exp(-0.5 * ((u - vb) ** 2.0 + v ** 2.0))
        maxwell2 = factor * cp.exp(-0.5 * ((u + vb) ** 2.0 + v ** 2.0))
        maxwell = 0.5 * (maxwell1 + maxwell2)
        self.arr_nodal = cp.tensordot(ix, cp.tensordot(iy, maxwell, axes=0), axes=0)
        # cartesian gradient
        grad_u, grad_v = (-(u - vb) * maxwell1 - (u + vb) * maxwell2) / 2, -v * maxwell

        # perturbation
        guess_r, guess_i = 0, 1
        for i in [2, 3, 4]:  # k_parallel modes
            for j in [0, 1, 2, 3, 4, 5]:  # k_perp modes
                phase_shift1 = 2 * cp.pi * cp.random.random(1)[0]
                phase_shift2 = 2 * cp.pi * cp.random.random(1)[0]
                # phase_shift3 = 2 * cp.pi * cp.random.random(1)[0]
                # phase_shift4 = 2 * cp.pi * cp.random.random(1)[0]
                # get eigenvalue
                k_x, k_y = grid.x.wavenumbers[grid.x.zero_idx + i], grid.y.wavenumbers[grid.y.zero_idx + j]
                k = np.sqrt(k_x ** 2.0 + k_y ** 2.0)
                phi = np.arctan2(k_y, k_x)
                solution = opt.root(dispersion.dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                    args=(k, vb, phi), jac=dispersion.jacobian_fsolve, tol=1.0e-10)
                guess_r, guess_i = solution.x
                print(solution.x * k)
                eigenfunction1 = 1.0e-2 * (k_x * grad_u + k_y * grad_v) / ((guess_r + 1j * guess_i) * k -
                                                                           k_x * u - k_y * v) / k
                eigenfunction2 = 1.0e-2 * (k_x * grad_u - k_y * grad_v) / ((guess_r + 1j * guess_i) * k -
                                                                           k_x * u + k_y * v) / k
                # eigenfunction3 = 1.0e-2 * (-k_x * grad_u + k_y * grad_v) / ((guess_r + 1j * guess_i) * k +
                #                                                            k_x * u - k_y * v) / k
                # eigenfunction4 = 1.0e-2 * (-k_x * grad_u - k_y * grad_v) / ((guess_r + 1j * guess_i) * k +
                #                                                            k_x * u + k_y * v) / k
                self.arr_nodal += cp.real(cp.tensordot(cp.exp(1j * (k_x * x + k_y * y + phase_shift1)),
                                                       eigenfunction1, axes=0))
                self.arr_nodal += cp.real(cp.tensordot(cp.exp(1j * (k_x * x - k_y * y + phase_shift2)),
                                                       eigenfunction2, axes=0))
                # self.arr_nodal += cp.real(cp.tensordot(cp.exp(1j * (-k_x * x + k_y * y + phase_shift3)),
                #                                        eigenfunction3, axes=0))
                # self.arr_nodal += cp.real(cp.tensordot(cp.exp(1j * (-k_x * x - k_y * y + phase_shift4)),
                #                                        eigenfunction4, axes=0))

                # self.arr_nodal += 0.01 * cp.tensordot(cp.sin((i+1) * grid.x.fundamental * grid.x.device_arr +
                #                                              phase_shift), cp.tensordot(iy, maxwell, axes=0), axes=0)
                # self.arr_nodal += 0.01 * cp.tensordot(ix,
                #                                       cp.tensordot(cp.sin((j+1) * grid.y.fundamental *
                #                                                           grid.y.device_arr + phase_shift),
                #                                            maxwell, axes=0), axes=0)

        self.fourier_transform()

    def initialize_maxwellian(self, grid):
        # space indicators
        ix, iy = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.y.device_arr)

        # velocity indicators
        iu, iv = cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        u, v = cp.tensordot(grid.u.device_arr, iv, axes=0), cp.tensordot(iu, grid.v.device_arr, axes=0)

        # set basic-ass maxwellian
        factor = 1 / (2 * cp.pi)
        maxwell = factor * cp.exp(-0.5 * (u ** 2.0 + v ** 2.0))
        self.arr_nodal = cp.tensordot(ix, cp.tensordot(iy, maxwell, axes=0), axes=0)

        # perturb
        # self.arr_nodal += 0.01 * cp.tensordot(cp.sin(grid.x.fundamental * grid.x.device_arr),
        #                                       cp.tensordot(cp.sin(grid.y.fundamental * grid.y.device_arr),
        #                                                    maxwell, axes=0), axes=0)
        self.arr_nodal += 0.01 * cp.tensordot(cp.sin(grid.x.fundamental * grid.x.device_arr),
                                              cp.tensordot(iy, maxwell, axes=0), axes=0)
        self.fourier_transform()


def trapz(f, dx, dy):
    """ Custom trapz routine using cupy """
    sum_y = cp.sum(f[:, :-1] + f[:, 1:], axis=1) * dy / 2.0
    return cp.sum(sum_y[:-1] + sum_y[1:]) * dx / 2.0
