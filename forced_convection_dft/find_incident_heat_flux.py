import numpy as np
from scipy.optimize import minimize
from dft_inverse_code import dft_models, heat_transfer_coefficients
from forced_convection_dft.correlations import PlateCorrelations, SphereCorrelations
import matplotlib.pyplot as plt


class HeatFluxFinder:
    def __init__(self, temperature_front, temperature_back, temperature_thermocouple, ambient_temperature,
                 temperature_initial, temperature_surroundings, time_vector, thermocouple_diameter,
                 thermocouple_absorptivity, thermocouple_emissivity, thermocouple_heat_capacity, thermocouple_density,
                 constant_velocity=True, plate_side_length=.0762):
        self.temperature_front = temperature_front
        self.temperature_back = temperature_back
        self.temperature_thermocouple = temperature_thermocouple
        self.temperature_initial = temperature_initial
        self.ambient_temperature = ambient_temperature
        self.time_vector = time_vector
        self.plate_side_length = plate_side_length
        self.thermocouple_diameter = thermocouple_diameter
        self.thermocouple_emissivity = thermocouple_emissivity
        self.temperature_surroundings = temperature_surroundings
        self.thermocouple_density = thermocouple_density
        self.thermocouple_heat_capacity = thermocouple_heat_capacity
        self.thermocouple_absorptivity = thermocouple_absorptivity
        self.constant_velocity = constant_velocity
        self.velocity = None
        self.q_inc = None
        self.h_front_dft, self.h_back_dft, self.h_tc = None, None, None

    def find_q_inc(self):
        velocity = self.recover_velocity()
        self.h_front_dft, self.h_back_dft, self.h_tc = self.find_heat_transfer_coefficients(velocity)
        q_inc = self.find_dft_q_inc(self.h_front_dft, self.h_back_dft)
        return q_inc

    def recover_velocity(self):
        if self.constant_velocity:
            starting_velocity = 15
            params = {'maxiter': 1e6}
            bounds = [(1e-3, np.inf)]
        else:
            starting_velocity = 15 * np.ones(self.time_vector.shape[0])
            params = {'ftol': 1}
            bounds = [(1e-3, np.inf)] * self.time_vector.shape[0]
        self.velocity = minimize(self.minimize_error, starting_velocity, method='powell', options=params,
                                 bounds=bounds).x
        return self.velocity

    def find_thermocouple_q_inc(self, h_tc):
        convective_heat_flux = h_tc * (self.temperature_thermocouple - self.ambient_temperature)
        area = 4 * np.pi * (self.thermocouple_diameter / 2) ** 2
        volume = 4 / 3 * np.pi * (self.thermocouple_diameter / 2) ** 3
        emitted_heat_flux = self.thermocouple_emissivity * 5.67e-8 * (
                self.temperature_thermocouple ** 4 - self.temperature_surroundings ** 4)
        energy_stored = (self.thermocouple_density * self.thermocouple_heat_capacity * volume) * np.gradient(
            self.temperature_thermocouple, self.time_vector)
        incident_heat_flux = (
                                     energy_stored / area + convective_heat_flux + emitted_heat_flux) / self.thermocouple_absorptivity
        return incident_heat_flux / 1e3

    def find_dft_q_inc(self, coefficient_front, coefficient_back):
        q_inc_plate = dft_models.one_dim_conduction(T_f=self.temperature_front, T_b=self.temperature_back,
                                                    time=self.time_vector,
                                                    h_f=coefficient_front, h_b=coefficient_back,
                                                    model='one_d_conduction', Kelvin=True,
                                                    T_inf=self.ambient_temperature,
                                                    T_sur=self.temperature_surroundings).q_inc
        return q_inc_plate

    def find_heat_transfer_coefficients(self, velocity):
        h_front_dft = PlateCorrelations(surface_temperature=self.temperature_front,
                                        ambient_temperature=self.ambient_temperature,
                                        characteristic_length=self.plate_side_length).find_forced_convection_heat_transfer_coefficient(
            velocity)
        h_back_dft = heat_transfer_coefficients.natural_convection(
            (self.temperature_back + self.ambient_temperature).reshape(-1, 1) / 2, Kelvin=True,
            T_infty=self.ambient_temperature, L_ch=self.plate_side_length).custom(0.65, 0.25).flatten()
        h_tc = SphereCorrelations(
            surface_temperature=self.temperature_thermocouple,
            ambient_temperature=self.ambient_temperature,
            characteristic_length=self.thermocouple_diameter).find_forced_convection_heat_transfer_coefficient(velocity)
        return h_front_dft, h_back_dft, h_tc

    def minimize_error(self, velocity):
        h_front_dft, h_back_dft, h_tc = self.find_heat_transfer_coefficients(velocity)
        q_inc_tc = self.find_thermocouple_q_inc(h_tc)
        q_inc_dft = self.find_dft_q_inc(h_front_dft, h_back_dft)
        return np.abs(q_inc_dft - q_inc_tc).sum()
