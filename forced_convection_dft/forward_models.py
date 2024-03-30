from .correlations import PlateCorrelations, SphereCorrelations
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from dft_inverse_code.dft_properties import ceramic_fiber, stainless_steel
from dft_inverse_code.heat_transfer_coefficients import natural_convection


class ForwardModelThermocouple:
    def __init__(self, time, q_inc, characteristic_velocity, temp_init,
                 temp_amb, temp_surr, emissivity, absorb, diameter, c_p,
                 rho):
        self.time = time
        self.q_inc = q_inc
        self.characteristic_velocity = characteristic_velocity
        self.temp_init = temp_init
        self.temp_amb = temp_amb
        self.temp_surr = temp_surr
        self.epsilon = emissivity
        self.absorb = absorb
        self.diameter = diameter
        self.c_p = c_p
        self.rho = rho
        self.volume = 4 / 3 * np.pi * (self.diameter / 2) ** 3
        self.area = 4 * np.pi * (self.diameter / 2) ** 2
        self.sig = 5.67e-8

    def run(self):
        n_time = len(self.time)
        delta_time = self.time[1] - self.time[0]
        tc_temp = np.zeros(n_time)
        tc_temp[0] = self.temp_init
        for t in tqdm(range(n_time - 1)):
            h = SphereCorrelations(
                surface_temperature=tc_temp[t],
                ambient_temperature=self.temp_amb,
                characteristic_length=self.diameter).find_forced_convection_heat_transfer_coefficient(
                self.characteristic_velocity[t])
            d_temp_d_time = self._calculate_d_temperature_d_time(tc_temp[t],
                                                                 h,
                                                                 self.q_inc[t],
                                                                 self.temp_amb)
            tc_temp[t + 1] = self._update_temperature(
                tc_temp[t], d_temp_d_time, delta_time)
        return tc_temp

    @staticmethod
    def _update_temperature(old_temp, d_temp_d_time, delta_time):
        new_temp = old_temp + d_temp_d_time * delta_time
        return new_temp

    def _calculate_d_temperature_d_time(self, tc_temp, h, q_inc,
                                        temp_amb):
        incident_heat_flux_minus_reflected = self.absorb * q_inc
        convective_heat_flux = h * (tc_temp - temp_amb)
        emitted_heat_flux = self.epsilon * self.sig * (
                tc_temp ** 4 - self.temp_surr ** 4)
        d_temperature_d_time = 1 / (self.rho * self.c_p * self.volume) * self.area * (
                incident_heat_flux_minus_reflected - convective_heat_flux - emitted_heat_flux)
        return d_temperature_d_time


class ForwardModelDft:
    def __init__(self, n_nodes, time, insul_thickness, plate_thickness, q_inc,
                 characteristic_velocity, temp_init, temp_amb, temp_surr,
                 dft_plate_length):
        self.n_nodes = n_nodes
        self.time = time
        self.n_steps = len(time)
        self.insul_thickness = insul_thickness
        self.plate_thickness = plate_thickness
        self.q_inc = q_inc
        self.characteristic_velocity = characteristic_velocity
        self.temp_init = temp_init
        self.temp_amb = temp_amb
        self.temp_surr = temp_surr
        self.sig = 5.67e-8
        self.delta_time = self.time[1] - self.time[0]
        self.plate_side_length = dft_plate_length
        self._initialize_arrays()

    def _find_insulation_properties(self, slab_thickness):
        insulation = ceramic_fiber(self.temp_init, Kelvin=True)
        diff = insulation.alpha
        k = insulation.k
        fourier_no = self.delta_time * diff / slab_thickness ** 2
        return fourier_no, k

    def _initialize_arrays(self):
        self.temp_f, self.temp_b = np.ones((2, self.n_steps)) * self.temp_init
        self.temp_ins = np.ones((self.n_steps, self.n_nodes)) * self.temp_init
        self.q_conv_f, self.q_conv_b, self.q_cond_f, self.q_cond_b = np.zeros((4, self.n_steps))
        self.h_f, self.h_b = np.zeros((2, self.n_steps))
        self.h_f[0], self.h_b[0] = self._find_heat_transfer_coefficients(
            self.temp_f[0], self.temp_b[0],
            self.characteristic_velocity[0])

    @staticmethod
    def _find_dft_properties(front_temperature, back_temperature):
        rcp_f = stainless_steel(front_temperature, Kelvin=True).rCp
        rcp_b = stainless_steel(back_temperature, Kelvin=True).rCp
        eps_f = stainless_steel(front_temperature, Kelvin=True).epsilon
        eps_b = stainless_steel(back_temperature, Kelvin=True).epsilon
        return rcp_f, rcp_b, eps_f, eps_b

    def _calculate_d_temp_d_time_plate(self, temp_plate, adjacent_slab_temp,
                                       k_ins, slab_thickness, rcp,
                                       epsilon, q_inc, h):
        q_ins_minus_ref = epsilon * q_inc
        q_conv = h * (temp_plate - self.temp_amb)
        q_emit = epsilon * self.sig * (temp_plate ** 4 - self.temp_surr ** 4)
        q_cond = k_ins / slab_thickness * (
                temp_plate - adjacent_slab_temp)
        d_temp_d_time = 1 / (self.plate_thickness * rcp) * (
                q_ins_minus_ref - q_conv - q_emit - q_cond)
        return q_conv, q_cond, d_temp_d_time

    @staticmethod
    def _update_insulation_temperature(temperature_slabs, temp_f,
                                       temp_b, fourier_number, equation_matrix):
        new_slab_temperatures = equation_matrix @ temperature_slabs
        new_slab_temperatures[0] = new_slab_temperatures[0] + fourier_number * temp_f
        new_slab_temperatures[-1] = new_slab_temperatures[-1] + fourier_number * temp_b
        new_slab_temperatures = new_slab_temperatures + temperature_slabs
        return new_slab_temperatures

    def _find_heat_transfer_coefficients(self, temp_f, temp_b, velocity):
        front_heat_transfer_coefficient = PlateCorrelations(
            temp_f,
            self.temp_amb,
            self.plate_side_length).find_forced_convection_heat_transfer_coefficient(
            velocity)
        back_heat_transfer_coefficient = natural_convection(
            (temp_b + self.temp_amb).reshape(-1, 1) / 2, Kelvin=True,
            T_infty=self.temp_amb, L_ch=self.plate_side_length).custom(0.65, 0.25)
        return front_heat_transfer_coefficient, back_heat_transfer_coefficient

    def _update_plate_temps(self, t, d_temperature_d_time_front,
                            d_temperature_d_time_back):
        self.temp_f[t + 1] = self.temp_f[
                                 t] + d_temperature_d_time_front * self.delta_time
        self.temp_b[t + 1] = self.temp_b[
                                 t] + d_temperature_d_time_back * self.delta_time

    def _update_heat_transfer_coefficient(self, time_step_number):
        self.h_f[time_step_number + 1], self.h_b[
            time_step_number + 1] = self._find_heat_transfer_coefficients(
            self.temp_f[time_step_number], self.temp_b[time_step_number],
            self.characteristic_velocity[time_step_number])

    def run(self):
        number_time_steps = len(self.time)
        slab_thickness = self.insul_thickness / (self.n_nodes + 1)
        insul_fourier, insul_k = self._find_insulation_properties(
            slab_thickness)
        insul_matrix = self._construct_insulation_matrix(insul_fourier)
        for t in tqdm(range(number_time_steps - 1)):
            rcp_f, rcp_b, eps_f, eps_b = self._find_dft_properties(
                self.temp_f[t], self.temp_b[t])
            self.q_conv_f[t], self.q_cond_f[t], d_temp_d_time_f = self._calculate_d_temp_d_time_plate(self.temp_f[t],
                                                                                                      self.temp_ins[
                                                                                                          t, 0],
                                                                                                      insul_k,
                                                                                                      slab_thickness,
                                                                                                      rcp_f, eps_f,
                                                                                                      self.q_inc[t],
                                                                                                      self.h_f[t])
            self.q_conv_b[t], self.q_cond_b[t], d_temp_d_time_b = self._calculate_d_temp_d_time_plate(self.temp_b[t],
                                                                                                      self.temp_ins[
                                                                                                          t, -1],
                                                                                                      insul_k,
                                                                                                      slab_thickness,
                                                                                                      rcp_f, eps_f,
                                                                                                      0,
                                                                                                      self.h_b[t])
            self.temp_ins[t + 1] = self._update_insulation_temperature(self.temp_ins[t], self.temp_f[t], self.temp_b[t],
                                                                       insul_fourier, insul_matrix)
            self._update_plate_temps(t, d_temp_d_time_f,
                                     d_temp_d_time_b)
            self._update_heat_transfer_coefficient(t)
        return self.temp_f, self.temp_b, self.temp_ins, self.h_f, self.h_b

    def _construct_insulation_matrix(self, fourier_number):
        k = np.array(
            [np.ones(self.n_nodes - 1) * fourier_number,
             np.ones(self.n_nodes) * - 2 * fourier_number,
             np.ones(self.n_nodes - 1) * fourier_number])
        offset = [-1, 0, 1]
        matrix = diags(k, offset).toarray()
        return matrix
