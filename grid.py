import numpy as np
import cupy as cp
import basis as b
import matplotlib.pyplot as plt
import scipy.special as sp


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """
    def __init__(self, low, high, elements, real_freqs=False):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.fundamental = 2.0 * np.pi / self.length
        if real_freqs:
            self.modes = elements // 2 + 1  # Nyquist frequency
            self.wavenumbers = self.fundamental * np.arange(self.modes)
            # de-aliasing parameters
            self.zero_idx = 0
        else:
            self.half_modes = elements // 2
            self.wavenumbers = self.fundamental * np.arange(-self.half_modes, self.half_modes)
            self.modes = self.wavenumbers.shape[0]
            self.zero_idx = self.half_modes
        # symmetry-independent spectral grid properties
        self.pad_width = self.modes // 3 + 1  # orszag two-thirds rule
        # self.device_modes = cp.arange(self.modes)
        self.device_wavenumbers = cp.array(self.wavenumbers)
        # for hyper-viscosity
        self.device_wavenumbers_fourth = self.device_wavenumbers ** 4.0
        self.device_wavenumbers_fourth[cp.absolute(self.device_wavenumbers) < 0.5] = 0

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)

        self.element_idxs = np.arange(self.elements)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        self.dx_grid = None
        # self.create_triple_grid(lows=np.array([self.low, -8, 8]),
        #                         highs=np.array([-8, 8, self.high]),
        #                         elements=np.array([2, 11, 2]))
        self.create_triple_grid(lows=np.array([self.low, -7.5, 7.5]),
                                highs=np.array([-7.5, 7.5, self.high]),
                                elements=np.array([2, 10, 2]))
        if self.dx_grid is None:
            self.dx_grid = self.dx * cp.ones(self.elements)

        # jacobian
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()
        self.min_dv = cp.amin(self.dx_grid)

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # create monotonic grid for plotting
        self.monogrid = None
        self.initialize_monogrid()

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def create_triple_grid(self, lows, highs, elements):
        """ Build a three-segment grid, each evenly-spaced """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        dxs = (highs - lows) / elements
        xl0 = np.linspace(lows[0], highs[0] - dxs[0], num=elements[0])
        xl1 = np.linspace(lows[1], highs[1] - dxs[1], num=elements[1])
        xl2 = np.linspace(lows[2], highs[2] - dxs[2], num=elements[2])
        # construct coordinates
        self.arr = np.zeros((elements[0] + elements[1] + elements[2], self.order))
        for i in range(elements[0]):
            self.arr[i, :] = xl0[i] + dxs[0] * nodes_iso
        for i in range(elements[1]):
            self.arr[elements[0] + i, :] = xl1[i] + dxs[1] * nodes_iso
        for i in range(elements[2]):
            self.arr[elements[0] + elements[1] + i, :] = xl2[i] + dxs[2] * nodes_iso
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])
        self.dx_grid = self.device_arr[:, -1] - self.device_arr[:, 0]

    def integrate(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], function, axes=([0, 1], idx))

    def second_moment(self, function, idx):
        return cp.tensordot(self.global_quads / self.J[:, None], cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                                             function),
                            axes=([0, 1], idx))

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                              thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))

    def initialize_monogrid(self):
        self.monogrid = np.zeros(self.elements * (self.order - 1) + 1)
        for i in range(self.elements):
            self.monogrid[i*(self.order-1):(i+1)*(self.order-1)] = self.arr[i, :-1]
        self.monogrid[-1] = self.arr[-1, -1]


class PhaseSpace:
    """ In this experiment, PhaseSpace consists of equispaced spatial nodes and a
    LGL tensor-product grid in truncated velocity space """

    def __init__(self, lows, highs, elements, order, charge_sign):
        # Grids
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.y = SpaceGrid(low=lows[1], high=highs[1], elements=elements[1], real_freqs=True)
        self.u = VelocityGrid(low=lows[2], high=highs[2], elements=elements[2], order=order)
        self.v = VelocityGrid(low=lows[3], high=highs[3], elements=elements[3], order=order)
        self.charge_sign = charge_sign

        # Square quantities
        self.v_mag_sq = (self.u.device_arr[:, :, None, None] ** 2.0 +
                         self.v.device_arr[None, None, :, :] ** 2.0)
        self.k_sq = (self.x.device_wavenumbers[:, None] ** 2.0 + self.y.device_wavenumbers[None, :] ** 2.0)

    def moment0(self, variable):
        return self.u.integrate(
            function=self.v.integrate(
                function=variable,
                idx=[4, 5]
            ),
            idx=[2, 3]
        )

    def moment2(self, variable):
        integrand = self.v_mag_sq[None, None, :, :, :, :] * variable
        return self.u.integrate(
            function=self.v.integrate(
                function=integrand,
                idx=[4, 5]
            ),
            idx=[2, 3]
        )

    def moment1_u(self, variable):
        integrand = self.u.device_arr[None, None, :, :, None, None] * variable
        return self.u.integrate(
            function=self.v.integrate(
                function=integrand,
                idx=[4, 5]
            ),
            idx=[2, 3]
        )

    def moment1_v(self, variable):
        integrand = self.v.device_arr[None, None, None, None, :, :] * variable
        return self.u.integrate(
            function=self.v.integrate(
                function=integrand,
                idx=[4, 5]
            ),
            idx=[2, 3]
        )
