import numpy as np
import scipy.optimize as opt
import dft_inverse_code.heat_transfer_coefficients as dft


class PlateCorrelations:
    """
    correlations
    """

    def __init__(self, surface_temperature, ambient_temperature, characteristic_length):
        """
        Parameters
        ----------
            surface_temperature: float, array like
                Surface temperature
            characteristic_length: float (optional)
                Characteristic length for determining heat transfer coefficients
             """

        self.eval_temperature = (surface_temperature + ambient_temperature) / 2
        self.ambient_temperature = ambient_temperature
        self.characteristic_length = characteristic_length
        self.kinematic_viscosity, _, self.thermal_conductivity, self.prandtl_number, _ = self.calculate_air_properties(
            self.eval_temperature)

    def calculate_air_properties(self, temperature):
        """
        returns air properties
        """
        kinematic_viscosity = dft.air_props(temperature, Kelvin=True).nu
        alpha = dft.air_props(temperature, Kelvin=True).alpha
        thermal_conductivity = dft.air_props(temperature, Kelvin=True).k
        prandtl_number = kinematic_viscosity / alpha
        density = self.find_density(temperature)
        return kinematic_viscosity, alpha, thermal_conductivity, prandtl_number, density

    @staticmethod
    def find_density(temperature):
        """
        Parameters
        ----------
            temperature: scalar or vector
                Temperature at which to find density

        Returns
        ----------
            rho: scalar or vector
                Density
        """

        R_air = 287.05
        P_atm = 101325
        density = P_atm / (R_air * temperature)
        return density

    def find_forced_convection_heat_transfer_coefficient(self, characteristic_velocity):
        """
        Parameters
        ----------
            characteristic_velocity: float or array like
                Flow velocity
        Returns
        ----------
            float or array like
                Heat transfer coefficient
        """
        reynolds_number = characteristic_velocity * self.characteristic_length / self.kinematic_viscosity
        nusselt_number = 0.664 * reynolds_number ** 0.5 * self.prandtl_number ** (1 / 3)  # Nu = hL/k
        heat_transfer_coefficient = nusselt_number * self.thermal_conductivity / self.characteristic_length
        return heat_transfer_coefficient

    def find_mixed_convection_heat_transfer_coefficient(self, characteristic_velocity):
        """
        Parameters
        ----------
            characteristic_velocity: float or array like
                velocity

        Returns
        ----------
            float or array like
                Mixed convection heat transfer coefficient
        """

        forced_heat_transfer_coefficient = self.find_forced_convection_heat_transfer_coefficient(
            characteristic_velocity)
        nusselt_forced = forced_heat_transfer_coefficient * self.characteristic_length / self.kinematic_viscosity
        free_heat_transfer_coefficient = dft.natural_convection(self.eval_temperature, Kelvin=True,
                                                                T_infty=self.ambient_temperature).custom(0.65, 0.25)
        nusselt_free = free_heat_transfer_coefficient * self.characteristic_length / self.kinematic_viscosity
        nusselt_mixes = (nusselt_forced ** 3 + nusselt_free ** 3) ** (1 / 3)
        mixed_heat_transfer_coefficient = nusselt_mixes * self.thermal_conductivity / self.characteristic_length
        return mixed_heat_transfer_coefficient


class SphereCorrelations(PlateCorrelations):
    def __init__(self, surface_temperature, ambient_temperature, characteristic_length):
        super().__init__(surface_temperature, ambient_temperature, characteristic_length)
        self.eval_temperature = self.ambient_temperature
        self.ambient_temperature = ambient_temperature
        self.characteristic_length = characteristic_length
        self.surface_temperature = surface_temperature
        self.kinematic_viscosity, _, self.thermal_conductivity, self.prandtl_number, _ = self.calculate_air_properties(
            self.eval_temperature)
        self.kinematic_viscosity_s, _, _, _, _ = self.calculate_air_properties(
            self.surface_temperature)

    def find_forced_convection_heat_transfer_coefficient(self, characteristic_velocity):
        """
        Parameters
        ----------
            characteristic_velocity: float or array like
                Flow velocity

        Returns
        ----------
            h: float or array like
                Heat transfer coefficient
        """
        reynolds_number = characteristic_velocity * self.characteristic_length / self.kinematic_viscosity
        nusselt_number = 2 + (
                0.4 * reynolds_number ** 0.5 + 0.06 * reynolds_number ** (2 / 3)) * self.prandtl_number ** 0.4 * (
                                 self.kinematic_viscosity / self.kinematic_viscosity_s) ** 0.25
        heat_transfer_coefficient = nusselt_number * self.thermal_conductivity / self.characteristic_length
        return heat_transfer_coefficient
