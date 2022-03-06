import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd

vt_c = 0.01
sq2 = 2 ** 0.5
tu = 1
tv = 1
e_sq = 1 - (tu/tv)**2.0


def electromagnetic_anisotropic_dispersion(k, z, phi):
    a = e_sq * np.sin(phi) * np.cos(phi) / (1 - e_sq * (np.cos(phi) ** 2))
    b = (e_sq * np.sin(phi) * np.cos(phi)) ** 2 / ((1 - e_sq * (np.cos(phi) ** 2)) * (1 - e_sq * (np.sin(phi) ** 2)))
    b_factor = np.sqrt((1-b)/(1+b))

    # v_parallel
    v_parallel = tu / np.sqrt(1 - e_sq * (np.sin(phi) ** 2))
    v_perp = tu / np.sqrt(1 - e_sq * (np.cos(phi) ** 2))
    z_norm = np.sqrt(0.5 * (1 + b)) * z / v_parallel

    # Integrals
    i1 = -0.25 * b_factor * (pd.Ztripleprime(z_norm) + 6 * pd.Zprime(z_norm))
    i2 = 0.25 * a * b_factor * (pd.Ztripleprime(z_norm) + 8 * pd.Zprime(z_norm))
    i3 = -0.25 * b_factor * ((a ** 2) * (pd.Ztripleprime(z_norm) + 2 * pd.Zprime(z_norm)) +
                             2 * (1+b) * (v_perp / v_parallel) ** 2 * pd.Zprime(z_norm))

    # Tensor components
    t11 = 1 - (1 - i1) / ((k * z) ** 2)
    t12 = i2 / ((k * z) ** 2)
    t22 = 1 - 1 / ((vt_c * z) ** 2) - (1 - i3) / ((k * z) ** 2)

    return t11 * t22 - t12 ** 2


# phase velocities
zr = np.linspace(-5, 5, num=400)
zi = np.linspace(-5, 2, num=200)
z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)
ZR, ZI = np.meshgrid(zr, zi, indexing='ij')


k_x = 0.01
k_y = 0.75
k = np.sqrt(k_x**2.0 + k_y**2.0)
phi = np.arctan2(k_y, k_x)
ep = electromagnetic_anisotropic_dispersion(k, z, phi)

plt.figure()
plt.contour(ZR, ZI, np.real(ep), 0, colors='r', linewidths=3)
plt.contour(ZR, ZI, np.imag(ep), 0, colors='g', linewidths=3)
plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
plt.grid(True), plt.tight_layout()
plt.show()
