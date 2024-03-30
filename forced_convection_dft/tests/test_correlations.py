import unittest
from forced_convection_dft.correlations import PlateCorrelations, SphereCorrelations
import numpy as np


class TestStringMethods(unittest.TestCase):
    @staticmethod
    def get_inputs():
        """
        get inputs
        """
        n_points = 50
        surface_temperature = np.ones(n_points) + np.random.normal(size=n_points) * 100 + 273
        ambient_temperature = np.ones(n_points) + np.random.normal(size=n_points) * 20 + 273
        characteristic_velocity = np.ones(n_points) * 3 + np.random.normal(size=n_points)
        characteristic_length = 10e-2
        return surface_temperature, ambient_temperature, characteristic_velocity, characteristic_length

    def test_plate(self):
        surface_temperature, ambient_temperature, characteristic_velocity, characteristic_length = self.get_inputs()
        plate = PlateCorrelations(surface_temperature, ambient_temperature, characteristic_length)
        calculated_coefficient = plate.find_forced_convection_heat_transfer_coefficient(characteristic_velocity)
        recovered_velocity = plate.find_forced_convection_characteristic_velocity(calculated_coefficient)
        self.assertTrue(np.allclose(characteristic_velocity, recovered_velocity, atol=1e-2))

    def test_sphere(self):
        surface_temperature, ambient_temperature, characteristic_velocity, characteristic_length = self.get_inputs()
        sphere = SphereCorrelations(surface_temperature, ambient_temperature, characteristic_length)
        calculated_coefficient = sphere.find_forced_convection_heat_transfer_coefficient(characteristic_velocity)
        recovered_velocity = sphere.find_forced_convection_characteristic_velocity(calculated_coefficient)
        self.assertTrue(np.allclose(characteristic_velocity, recovered_velocity, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
