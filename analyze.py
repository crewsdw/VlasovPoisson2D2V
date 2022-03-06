import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import data
import cupy as cp
import matplotlib.pyplot as plt

# # Geometry and grid parameters
# elements, order = [22, 48, 15, 15], 12
#
# # Grid
# wavenumber_x = 0.2
# wavenumber_y = 0.02
# Geometry and grid parameters
elements, order = [40, 50, 14, 14], 8

# Grid
wavenumber_x = 0.05
wavenumber_y = 0.02
length_x, length_y = 2.0 * np.pi / wavenumber_x, 2.0 * np.pi / wavenumber_y
lows = np.array([-length_x / 2, -length_y / 2, -10, -10])
highs = np.array([length_x / 2, length_y / 2, 10, 10])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Read data
DataFile = data.Data(folder='two_stream\\', filename='test_feb9_12mode')
time_data, distribution_data, density_data, potential_data, total_eng, total_den = DataFile.read_file()
# print(distribution_data.shape)
min_f, max_f = np.amin(distribution_data), np.amax(distribution_data[0, :, :, :, :, :, :])
min_n, max_n = np.amin(density_data), np.amax(density_data)
min_p, max_p = np.amin(potential_data), np.amax(potential_data)
if min_f < 0:
    min_f = 0
print(min_f), print(max_f)

# Set up plotter
plotter = my_plt.Plotter2D(grid=grid)
v_plotter = my_plt.VelocityPlotter(grid=grid)

jump = 0
for idx, time in enumerate(time_data[jump:]):
    print('Data at time {:0.3e}'.format(time))
    # Unpack data, distribution, density
    distribution = var.Distribution(resolutions=elements, order=order)
    distribution.arr_nodal = cp.asarray(distribution_data[idx])
    distribution.moment0.arr_nodal = cp.asarray(density_data[idx])
    distribution.fourier_transform(), distribution.moment0.fourier_transform()
    # compute momentum
    distribution.compute_moment1(grid=grid)
    distribution.compute_moment1_magnitude()
    # compute vorticity
    vorticity = var.SpaceScalar(resolutions=[elements[0], elements[1]])
    vorticity.arr_spectral = 1j * (grid.x.device_wavenumbers[:, None] * distribution.moment1.arr_spectral[1, :, :] -
                                   grid.y.device_wavenumbers[None, :] * distribution.moment1.arr_spectral[0, :, :])
    vorticity.inverse_fourier_transform()
    # compute potential
    static_field = fields.Static(resolutions=[elements[0], elements[1]])
    static_field.poisson(distribution=distribution, grid=grid)
    # compute average distribution
    average_distribution = var.VelocityScalar(resolutions=[elements[2], elements[3]], order=order)
    distribution.set_modal_distribution(idx=[grid.x.zero_idx, grid.y.zero_idx], velocity_scalar=average_distribution)

    v_plotter.velocity_contourf(dist_slice=average_distribution.arr_nodal, cb_lim=[min_f, max_f],
                                save='figs\\average_distribution\\' + str(idx) + '.png')
    # plotter.spatial_scalar_plot(scalar=distribution.moment0, title='Density', spectrum=False)
    plotter.scalar_plot_interpolated(scalar=distribution.moment0, cb_lim=[min_n, max_n],
                                     title='Density', save='figs\\density\\' +
                                                           str(idx) + '.png')
    plotter.scalar_plot_interpolated(scalar=static_field.potential, cb_lim=[min_p, max_p],
                                     title='Potential', save='figs\\potential\\' +
                                                             str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=distribution.moment1_magnitude, cb_lim=[min_f, max_f],
    #                                  title='Momentum',
    #                                  save='figs\\momentum\\' + str(idx) + '.png')
    # plotter.scalar_plot_interpolated(scalar=vorticity, title='Vorticity', save='figs\\vorticity\\' +
    #                                  str(idx) + '.png')
    plt.close('all')
    # plotter.show()

# plotter.show()
