import numpy as np
import scipy.special as sp


def Z_uniform(z):
    out = np.zeros_like(z) + 0j
    z_large = z[np.abs(z) >= 15]
    z_small = z[np.abs(z) < 15]
    out[np.abs(z) < 15] = 1j * np.sqrt(np.pi) * np.exp(-z_small ** 2.0) * (1.0 + sp.erf(1j * z_small))
    # out[np.abs(z) < 15] = np.sqrt(np.pi) * 1j * sp.wofz(z_small)
    out[np.abs(z) >= 15] = Z_asymptotic(z_large)
    return out


def Z_asymptotic(z):
    sig = np.zeros_like(z)
    sig[np.imag(z) > 0] = 0
    sig[np.imag(z) == 0] = 1
    sig[np.imag(z) < 0] = 2

    return 1j * sig * np.sqrt(np.pi) * np.exp(-z ** 2.0) - (1.0 / z + 1.0 / (2.0 * (z ** 3.0)) +
                                                            3.0 / (4.0 * (z ** 5.0)) + 15.0 / (8.0 * (z ** 7.0)))


def Z_asymptotic_no_exp(z):
    return - (1.0 / z + 1.0 / (2.0 * (z ** 3.0)) + 3.0 / (4.0 * (z ** 5.0)) + 15.0 / (8.0 * (z ** 7.0)))


def a_cf(z, n):
    n -= 1
    if n == 0:
        return -z
    else:
        return 0.5 * n * (2 * n - 1)


def b_cf(z, n):
    n -= 1
    return -z ** 2.0 + 2 * n + 0.5


def Z_continued_fraction(z, terms):
    A = np.zeros((terms + 1, z.shape[0])) + 0j
    B = np.zeros((terms + 1, z.shape[0])) + 0j
    A[0, :], A[1, :] = 1, 0
    B[0, :], B[1, :] = 0, 1

    for n in range(1, terms):
        a, b = a_cf(z, n), b_cf(z, n)
        A[n + 1, :] = -1.0 * a * A[n - 1, :] + b * A[n, :]
        B[n + 1, :] = -1.0 * a * B[n - 1, :] + b * B[n, :]

    return np.divide(A[-1, :], B[-1, :])


def Z(z):
    y_cutoff = 4
    if type(z) is np.ndarray:
        out = np.zeros_like(z) + 0j
        # x, y = np.real(z), np.imag(z)

        out[np.imag(z) < y_cutoff] = Z_uniform(z[np.imag(z) < y_cutoff])
        out[np.imag(z) >= y_cutoff] = Z_continued_fraction(z[np.imag(z) >= y_cutoff], terms=10)
        return out
    if type(z) is np.complex128:
        z = np.array([z, 0 + 0j])
        if np.imag(z[0]) < y_cutoff:
            out = Z_uniform(z)
        else:
            out = Z_continued_fraction(z, terms=10)
        return out[0]


def Zprime(z):
    return -2 * (1 + z * Z(z))


def Zdoubleprime(z):
    return -2 * (Z(z) + z * Zprime(z))


def Ztripleprime(z):
    return -2.0 * (2 * Zprime(z) + z * Zdoubleprime(z))
