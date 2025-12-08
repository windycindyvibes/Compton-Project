import numpy as np


def LorentzBoost(old_momentum, beta):
    beta_x, beta_y, beta_z = beta
    beta_sqr = beta_x**2 + beta_y**2 + beta_z**2
    gamma = 1 / np.sqrt(1 - beta_x**2 - beta_y**2 - beta_z**2)
    old_energy = old_momentum[0]
    old_kvec = old_momentum[1:] / old_energy
    old_kx, old_ky, old_kz = old_kvec
    new_energy = (
        gamma * old_energy * (1 - old_kx * beta_x - old_ky * beta_y - old_kz * beta_z)
    )
    new_photon_momx = (
        (
            old_ky * beta_x * beta_y * (gamma - 1)
            + old_kx * (beta_y**2 + beta_z**2 + beta_x**2 * gamma)
            - beta_x
            * (
                old_kz * beta_z
                + (beta_x**2 + beta_y**2 + beta_z * (beta_z - old_kz)) * gamma
            )
        )
        * old_energy
    ) / beta_sqr
    new_photon_momy = (
        (
            old_ky * (beta_x**2 + beta_z**2 + beta_y**2 * gamma)
            + beta_y
            * (
                old_kx * beta_x * (gamma - 1)
                + old_kz * beta_z * (gamma - 1)
                - beta_sqr * gamma
            )
        )
        * old_energy
    ) / beta_sqr
    new_photon_momz = (
        (
            old_kz * (beta_x**2 + beta_y**2 + beta_z**2 * gamma)
            + beta_z
            * (
                old_kx * beta_x * (gamma - 1)
                + old_ky * beta_y * (gamma - 1)
                - beta_sqr * gamma
            )
        )
        * old_energy
    ) / beta_sqr
    return np.array([new_energy, new_photon_momx, new_photon_momy, new_photon_momz])


def ComptonScattering(photon_momentum, electron_velocity):
    fluid_gamma = electron_velocity[0]
    fluid_kvec = electron_velocity[1:] / np.linalg.norm(electron_velocity[1:])
    fluid_beta = np.sqrt(1.0 - fluid_gamma**-2)
    fluid_betavec = fluid_beta * fluid_kvec

    photon_momentum_1 = LorentzBoost(photon_momentum, fluid_betavec)

    # TASK: add Compton scattering
    # 1. P1 = photon_momentum_1 -- four-momentum of the photon (dimensionless) in the frame of the electron (fluid)
    # 2. E1 = photon_momentum_1[0] -- dimensionless energy of the photon in the frame of the electron (before the scattering)
    # 3. SCATTERING -> E2 -- energy of the photon after the scattering (depends on theta)
    #   3.1 theta is a random angle -> take randomly from 0 to pi
    #   3.2. phi -- angle in the other direction -> take randomly from 0 to 2 * pi
    #   3.3. find E2 using the compton formula (use theta)
    #   3.4. construct photon_momentum_2 -- four-momentum of the photon AFTER the scattering in the frame of the electron (use the direction from theta and phi)
    # 4. boost photon_momentum_2 to the lab frame (back from the electron frame)
    theta = np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    E2 = photon_momentum_1[0] / (1 + photon_momentum_1[0] * (1 - np.cos(theta)))

    k1 = photon_momentum[1:] / np.linalg.norm(photon_momentum[1:])
    r = np.random.rand(3)
    r /= np.linalg.norm(r)
    e2 = np.cross(k1, r)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(k1, e2)

    k2 = (
        k1 * np.cos(theta)
        + e2 * np.sin(theta) * np.sin(phi)
        + e3 * np.sin(theta) * np.cos(phi)
    )

    photon_momentum_2 = np.append([E2], E2 * k2)
    photon_momentum_lab_2 = LorentzBoost(photon_momentum_2, -fluid_betavec)

    return photon_momentum_lab_2
