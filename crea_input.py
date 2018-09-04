"""Crea i dati in input per il problema degli n-corpi"""

import math

import numpy as np
import tqdm


def polar_to_cartesian(rho: float, theta: float):
    """
    Convert from polar to cartesian coordinates.
    Theta is in radians.
    """
    return rho * math.cos(theta), rho * math.sin(theta)


def draw_coords(max_rho=1000, max_theta=360.0):
    """
    Draw the polar coordinates from a uniform distribution,
    with range (0, max_rho) for modulus and (0, max_theta)
    for the angle.
    """
    rho = max_rho * np.random.rand()
    theta_deg = max_theta * np.random.rand()
    theta_radians = math.pi * (theta_deg / 180.0)

    return rho, theta_radians


def get_velocity(rho, theta, max_rho):
    """
    Get the velocity as a function of rho and theta.
    Theta is in radians.
    """
    jitter = np.random.rand() * 0.15
    v_mod = (rho / max_rho) + jitter
    vx = v_mod * math.sin(theta)
    vy = v_mod * (-math.cos(theta))

    return vx, vy


def get_mass(rho, max_rho, max_mass):
    """
    Get the mass of a body as a function of its distance from the
    galactic centre. It is a linear relation, plus a jitter drawn
    from a uniform distribution.
    """
    jitter = np.random.rand() * (max_mass * 0.05)
    m = (rho / max_rho) * max_mass + jitter

    return m


def write_file(filename: str, data: np.array):
    """
    Write the contents of the numpy array to a file.
    """
    np.savetxt(filename, data)


if __name__ == '__main__':
    n_bodies = 65536
    x = np.zeros((n_bodies, 1))
    y = np.zeros((n_bodies, 1))
    vx = np.zeros((n_bodies, 1))
    vy = np.zeros((n_bodies, 1))
    m = np.zeros((n_bodies, 1))

    max_rho = 1e3
    max_mass = 1e2
    max_theta = 360.0

    for i in tqdm.tqdm(range(n_bodies)):
        rho, theta = draw_coords(max_rho, max_theta)
        x_i, y_i = polar_to_cartesian(rho, theta)
        vx_i, vy_i = get_velocity(rho, theta, max_rho)

        x[i] = x_i
        y[i] = y_i
        vx[i] = vx_i
        vy[i] = vy_i
        m[i] = get_mass(rho, max_rho, max_mass)

    write_file("./data/x.txt", x)
    write_file("./data/y.txt", y)
    write_file("./data/vx.txt", vx)
    write_file("./data/vy.txt", vy)
    write_file("./data/mass.txt", m)
