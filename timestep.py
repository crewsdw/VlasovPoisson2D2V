import numpy as np
import time as timer
import variables as var
import fields
import cupy as cp

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, dt, resolutions, order, steps, grid, phase_space_flux):
        self.x_res, self.y_res, self.u_res, self.v_res = resolutions
        self.resolutions = resolutions
        self.order = order
        self.dt = dt
        self.steps = steps
        self.phase_space_flux = phase_space_flux

        # RK coefficients
        self.rk_coefficients = np.array(nonlinear_ssp_rk_switch.get(3, "nothing"))

        # tracking arrays
        self.time = 0
        self.next_time = 0
        self.time_array = np.array([])
        self.electric_energy = np.array([])
        self.thermal_energy = np.array([])
        self.density_array = np.array([])

        # semi-implicit advection matrix
        self.implicit_x_advection_matrix = None
        self.implicit_y_advection_matrix = None
        self.build_advection_matrices(grid=grid, dt=0.5 * self.dt)  # applied twice, once on each side of nonlinear term

        # save-times
        self.save_times = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
                                    6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
                                    12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5,
                                    18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5,
                                    24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 0])

    def build_advection_matrices(self, grid, dt):
        """ Construct the global backward advection matrix """
        # x-directed velocity
        backward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] -
                                       0.5 * dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                       grid.u.translation_matrix[None, :, :, :])
        forward_advection_operator = (cp.eye(grid.u.order)[None, None, :, :] +
                                      0.5 * dt * -1j * grid.x.device_wavenumbers[:, None, None, None] *
                                      grid.u.translation_matrix[None, :, :, :])
        inv_backward_advection = cp.linalg.inv(backward_advection_operator)
        self.implicit_x_advection_matrix = cp.matmul(inv_backward_advection, forward_advection_operator)

        # y-directed advection
        backward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] -
                                       0.5 * dt * -1j * grid.y.device_wavenumbers[:, None, None, None] *
                                       grid.v.translation_matrix[None, :, :, :])
        forward_advection_operator = (cp.eye(grid.v.order)[None, None, :, :] +
                                      0.5 * dt * -1j * grid.y.device_wavenumbers[:, None, None, None] *
                                      grid.v.translation_matrix[None, :, :, :])
        inv_backward_advection = cp.linalg.inv(backward_advection_operator)
        self.implicit_y_advection_matrix = cp.matmul(inv_backward_advection, forward_advection_operator)

    def main_loop(self, distribution, static_field, grid, datafile):
        print('Beginning main loop')
        # Compute first two steps with ssp-rk3 and save fluxes
        # zero stage
        ps_flux0 = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                     field=static_field, grid=grid)

        # first step
        self.ssp_rk3(distribution=distribution, field=static_field, grid=grid)
        self.time += self.dt

        # first stage
        ps_flux1 = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                     field=static_field, grid=grid)

        # second step
        self.ssp_rk3(distribution=distribution, field=static_field, grid=grid)
        self.time += self.dt

        # store first two fluxes
        previous_phase_space_fluxes = [ps_flux1, ps_flux0]

        # Main loop
        t0, save_counter = timer.time(), 0
        for i in range(self.steps):
            previous_phase_space_fluxes = self.strang_split_adams_bashforth(
                distribution=distribution, field=static_field, grid=grid,
                previous_phase_space_fluxes=previous_phase_space_fluxes
            )
            self.time += self.dt

            if i % 10 == 0:
                self.time_array = np.append(self.time_array, self.time)
                static_field.poisson(distribution=distribution, grid=grid, invert=True)
                self.electric_energy = np.append(self.electric_energy,
                                                 static_field.compute_field_energy(grid=grid).get())
                self.thermal_energy = np.append(self.thermal_energy, distribution.total_thermal_energy(grid=grid).get())
                self.density_array = np.append(self.density_array, distribution.total_density(grid=grid).get())
                print(cp.amax(static_field.electric.arr_nodal))
                print('\nTook 10 steps, time is {:0.3e}'.format(self.time))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')

            if np.abs(self.time - self.save_times[save_counter]) < 6.0e-3:
                print('Reached save time at {:0.3e}'.format(self.time) + ', saving data...')
                distribution.inverse_fourier_transform()
                distribution.moment0.inverse_fourier_transform()
                datafile.save_data(distribution=distribution.arr_nodal.get(),
                                   density=distribution.moment0.arr_nodal.get(),
                                   potential=static_field.potential.arr_nodal.get(), time=self.time)
                save_counter += 1

        print('\nAll done at time is {:0.3e}'.format(self.time))
        print('Total steps were ' + str(self.steps))
        print('Time since start is {:0.3e}'.format((timer.time() - t0)))

    def ssp_rk3(self, distribution, field, grid):
        # Cut-off (avoid CFL advection instability as this is fully explicit)
        # cutoff = 50

        # Stage set-up
        phase_space_stage0 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)
        phase_space_stage1 = var.Distribution(resolutions=self.resolutions,
                                              order=self.order)

        # zero stage
        field.poisson(distribution=distribution, grid=grid)
        #
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=distribution,
                                                                             field=field,
                                                                             grid=grid)
        # skip cut-off for now
        # phase_space_rhs[grid.x.device_modes > cutoff, grid.y.device_modes > :, :, :, :] = 0
        #
        phase_space_stage0.arr_spectral = (distribution.arr_spectral +
                                           self.dt * phase_space_rhs)

        # first stage
        # field.poisson(distribution=phase_space_stage0, grid=grid)
        #
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=phase_space_stage0,
                                                                             field=field,
                                                                             grid=grid)
        #
        # phase_space_rhs[grid.x.device_modes > cutoff, :, :, :, :] = 0
        #
        phase_space_stage1.arr_spectral = (
                self.rk_coefficients[0, 0] * distribution.arr_spectral +
                self.rk_coefficients[0, 1] * phase_space_stage0.arr_spectral +
                self.rk_coefficients[0, 2] * self.dt * phase_space_rhs
        )

        # second stage
        # field.poisson(distribution=phase_space_stage1, grid=grid)
        #
        phase_space_rhs = self.phase_space_flux.semi_discrete_fully_explicit(distribution=phase_space_stage1,
                                                                             field=field,
                                                                             grid=grid)
        #
        # phase_space_rhs[grid.x.device_modes > cutoff, :, :, :, :] = 0
        #
        distribution.arr_spectral = (
                self.rk_coefficients[1, 0] * distribution.arr_spectral +
                self.rk_coefficients[1, 1] * phase_space_stage1.arr_spectral +
                self.rk_coefficients[1, 2] * self.dt * phase_space_rhs
        )

    def strang_split_adams_bashforth(self, distribution, field, grid, previous_phase_space_fluxes):
        # strang-split phase space advance
        # next_arr = cp.zeros_like(distribution.arr_spectral)

        # half crank-nicholson fractional advance advection step
        next_arr = cp.einsum('njkl,nmjlpq->nmjkpq',
                             self.implicit_x_advection_matrix, distribution.arr_spectral)
        next_arr = cp.einsum('mpqr,nmjkpr->nmjkpq',
                             self.implicit_y_advection_matrix, next_arr)

        # full explicit adams-bashforth nonlinear momentum flux step
        phase_space_rhs = self.phase_space_flux.semi_discrete_semi_implicit(distribution=distribution,
                                                                            field=field, grid=grid)
        next_arr += self.dt * (
            (23 / 12 * phase_space_rhs -
             4 / 3 * previous_phase_space_fluxes[0] +
             5 / 12 * previous_phase_space_fluxes[1]))

        # further half crank-nicholson fractional advection step
        next_arr = cp.einsum('njkl,nmjlpq->nmjkpq',
                             self.implicit_x_advection_matrix, next_arr)
        distribution.arr_spectral = cp.einsum('mpqr,nmjkpr->nmjkpq',
                                              self.implicit_y_advection_matrix, next_arr)

        # save fluxes
        previous_phase_space_fluxes = [phase_space_rhs, previous_phase_space_fluxes[0]]
        return previous_phase_space_fluxes
