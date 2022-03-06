import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes
import timestep as ts
import data

# Geometry and grid parameters
elements, order = [40, 50, 14, 14], 8

# Grid
wavenumber_x = 0.05
wavenumber_y = 0.02
length_x, length_y = 2.0 * np.pi / wavenumber_x, 2.0 * np.pi / wavenumber_y
lows = np.array([-length_x/2, -length_y/2, -11.5, -11.5])
highs = np.array([length_x/2, length_y/2, 11.5, 11.5])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Variables: distribution
distribution = var.Distribution(resolutions=elements, order=order)
# distribution.initialize_maxwellian(grid=grid)
distribution.initialize_two_stream(grid=grid, vb=3)
# static and dynamic fields
static_field = fields.Static(resolutions=[elements[0], elements[1]])
static_field.poisson(distribution=distribution, grid=grid)

# Plotter: check out IC
plotter = my_plt.Plotter2D(grid=grid)
plotter.spatial_scalar_plot(scalar=distribution.moment0, title='Density', spectrum=False)
plotter.spatial_scalar_plot(scalar=static_field.potential, title='Potential', spectrum=False)
plotter.show()

# Set up fluxes
flux = fluxes.DGFlux(resolutions=elements, order=order, grid=grid, nu=10.0)
flux.initialize_zero_pad(grid=grid)

# Time information
dt, stop_time = 1.5e-2, 30
steps = int(stop_time // dt) + 1


# Save data
datafile = data.Data(folder='two_stream\\', filename='test_feb9_12mode')
datafile.create_file(distribution=distribution.arr_nodal.get(),
                     density=distribution.moment0.arr_nodal.get(),
                     potential=static_field.potential.arr_nodal.get())

# Set up time-stepper
stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps, grid=grid, phase_space_flux=flux)
stepper.main_loop(distribution=distribution, static_field=static_field, grid=grid, datafile=datafile)

# Check it out
plotter.spatial_scalar_plot(scalar=distribution.moment0, title='Density', spectrum=False)
plotter.spatial_scalar_plot(scalar=static_field.potential, title='Potential', spectrum=False)

plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.electric_energy, y_axis='Electric energy',
                         log=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy, y_axis='Thermal energy')
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array, y_axis='Total density')
plotter.time_series_plot(time_in=stepper.time_array, series_in=(stepper.electric_energy + stepper.thermal_energy),
                         y_axis='Total energy', log=False)
plotter.show()
