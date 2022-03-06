import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd

vt_c = 0.3
sq2 = 2 ** 0.5


def electrostatic_dispersion(k, z, vb, phi):
    k_sq = k ** 2.0
    z_plus = (z - np.cos(phi) * vb) / sq2
    z_minus = (z + np.cos(phi) * vb) / sq2

    return 1 - 0.5 * (0.5 * pd.Zprime(z_plus) + 0.5 * pd.Zprime(z_minus)) / k_sq


def analytic_jacobian(k, z, vb, phi):
    k_sq = k ** 2.0
    z_plus = (z - np.cos(phi) * vb) / sq2
    z_minus = (z + np.cos(phi) * vb) / sq2

    return - 0.5 * (0.5 * pd.Zdoubleprime(z_plus) + 0.5 * pd.Zdoubleprime(z_minus)) / k_sq


def dispersion_fsolve(z, k, vb, phi):
    complex_z = z[0] + 1j * z[1]
    d = electrostatic_dispersion(k, complex_z, vb, phi)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, vb, phi):
    complex_z = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, complex_z, vb, phi)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


if __name__ == '__main__':
    # phase velocities
    zr = np.linspace(-4, 4, num=200)
    zi = np.linspace(-4, 4, num=200)
    z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)
    ZR, ZI = np.meshgrid(zr, zi, indexing='ij')

    k_x = 0.2
    k_y = 0.02
    k = np.sqrt(k_x**2.0 + k_y**2.0)
    phi = np.arctan2(k_y, k_x)
    ep = electrostatic_dispersion(k, z, 3, phi)

    plt.figure()
    plt.contour(ZR, ZI, np.real(ep), 0, colors='r', linewidths=3)
    plt.contour(ZR, ZI, np.imag(ep), 0, colors='g', linewidths=3)
    plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
    plt.grid(True), plt.tight_layout()
    plt.show()

    # Obtain some solutions
    kx = np.linspace(0.05, 0.75, num=100)
    ky = np.linspace(0, 0.5, num=100)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    sols = np.zeros_like(KX) + 0j
    guess_r, guess_i = 0, 1
    for idx, k_x in enumerate(kx):
        # guess_r, guess_i = 0.0, 1
        # if k_x < 0.01:
        #     continue
        if idx > 0:
            guess_r, guess_i = np.real(sols[idx-1, 0]), np.imag(sols[idx-1, 0])
        for idy, k_y in enumerate(ky):
            k = np.sqrt(k_x**2.0 + k_y**2.0)
            phi = np.arctan2(k_y, k_x)
            # if phi > 1.3:  # np.abs(phi - np.pi/2) < 3.0e-1:
            #     continue
            solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                                args=(k, 3, phi), jac=jacobian_fsolve, tol=1.0e-10)
            guess_r, guess_i = solution.x
            sols[idx, idy] = (guess_r + 1j * guess_i)


    # rsol = np.real(sols)
    # # cbr = np.linspace(np.amin(rsol), np.amax(rsol), num=100)
    # plt.figure()
    # plt.contourf(KX, KY, rsol)  # , cbr)
    # plt.xlabel(r'Wavenumber $k_x\lambda_D$'), plt.ylabel(r'Wavenumber $k_y\lambda_D$'), plt.tight_layout()
    # plt.colorbar()

    isol = np.imag(sols)
    cbi = np.linspace(-np.amax(isol), np.amax(isol), num=100)
    plt.figure()
    plt.contourf(KX, KY, isol, cbi, extend='both')
    plt.colorbar()
    plt.contour(KX, KY, isol, 0, colors='r')
    plt.xlabel(r'Parallel wavenumber $(\vec{k}\lambda_D)\cdot\hat{u}_d$')
    plt.ylabel(r'Perpendicular wavenumber $(\vec{k}\lambda_D)\times\hat{u}_d$')
    plt.title(r'Im($\zeta$)/$v_t$, $\Delta u = 6v_t$'), plt.tight_layout()
    plt.savefig('figs/ud3_phase_velocity.pdf')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    K = np.sqrt(KX**2 + KY**2)
    om = K*isol
    cbo = np.linspace(-np.amax(om), np.amax(om), num=100)
    plt.figure()
    plt.contourf(KX, KY, om, cbo, extend='both')
    plt.colorbar()
    plt.contour(KX, KY, om, 0, colors='r')
    plt.xlabel(r'Parallel wavenumber $(\vec{k}\lambda_D)\cdot\hat{u}_d$')
    plt.ylabel(r'Perpendicular wavenumber $(\vec{k}\lambda_D)\times\hat{u}_d$')
    plt.title(r'Growth rate Im($\omega$)/$\omega_p$, $\Delta u = 6v_t$'), plt.tight_layout()
    plt.savefig('figs/ud3_growth_rate.pdf')

    plt.show()

    # plt.figure()
    # plt.plot(k, k * np.real(sols), 'r', linewidth=3, label='Real')
    # plt.plot(k, k * np.imag(sols), 'g', linewidth=3, label='Imaginary')
    # plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Frequency $\omega / \omega_p$')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    plt.show()
