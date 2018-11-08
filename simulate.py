import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
import math


class LithiumIonBattery:
    def __init__(self):
        self.maximum_capacity = 2.2  # [Amps * Hours]
        self.discharged_voltage = 3.0  # [Volts]
        self.charged_voltage = 4.2  # [Volts]
        self.polarization_constant = 0.0139  # [Amps / Hours]


class Simulate:
    def __init__(self, **kwargs):
        # Simulation Configuration
        self.cycles = kwargs.get('cycles', 1)
        self.time_step = kwargs.get('time_step_mins', 0.5)/60.0  # [Hours]
        self.charging_current = kwargs.get('charging_current', -1.1)  # [Amps]
        self.discharging_current = kwargs.get('discharging_current', 1.1)  # [Amps]

        self.battery_specs = LithiumIonBattery()

        # Constants
        self.ambient_temperature = 20  # [Degrees celsius]
        self.cutoff_current = 0.05 * self.battery_specs.maximum_capacity  # 0.05C
        # TODO: These should be variable in the complex model
        self.charging_internal_resistance = 0.035  # [Ohms] Average
        self.discharging_internal_resistance = 0.030  # [Ohms] Average
        self.temperature = self.ambient_temperature  # [Degrees celsius]
        self.A = -0.314  # Exponential constant A
        self.B = 40.71  # Exponential constant B

        # Initial conditions
        # Short term variables
        self.time = 0  # Hours
        self.voltage = self.battery_specs.discharged_voltage  # Discharged
        self.current = self.charging_current  # [Amps]

        self.capacity = 0  # [Amps * Hours]
        self.SOC = self.capacity/self.battery_specs.maximum_capacity  # [Percent]
        self.SOP = self.voltage/self.battery_specs.charged_voltage  # [Percent]

        # Long term variables
        self.cycle = 0
        self.DOD = 1  # [Percent] Depth of Discharge
        self.SOH = 1  # [Percent] State of Health
        self.maximum_capacity = 2.2  # [Amps * Hours]

        self.time_t, self.voltage_t, self.current_t, self.capacity_t, self.SOC_t, self.SOP_t, \
            self.cycle_c, self.DOD_c, self.SOH_c, self.maximum_capacity_c = [[] for i in range(10)]

    def save_cycle(self):
        print(f'Saving cycle. State of Health: {self.SOH * 100}%')
        self.cycle_c.append(self.cycle)
        self.DOD_c.append(self.DOD)
        self.SOH_c.append(self.SOH)
        self.maximum_capacity_c.append(self.maximum_capacity)

    def save_state(self):
        self.time_t.append(self.time)
        self.voltage_t.append(self.voltage)
        self.current_t.append(self.current)
        self.capacity_t.append(self.capacity)
        self.SOC_t.append(self.SOC)
        self.SOP_t.append(self.SOP)

    def get_constant_voltage(self):
        return self.battery_specs.charged_voltage + random.gauss(0, 0.0005)

    def step(self):
        self.time += self.time_step
        self.capacity -= self.current * self.time_step
        self.SOC = self.capacity / self.battery_specs.maximum_capacity

    def V_INT(self, n_p, x):
        n = 1  # for Li-ion battery
        F = 96485.33289  # Faraday's Constant
        if n_p == 'p':  # p
            A_k = np.array([
                -33642.23,
                0.11,
                23506.89,
                -74679.26,
                14359.34,
                307849.79,
                85053.13,
                -1075148.06,
                2173.62,
                991586.68,
                283423.47,
                -163020.34,
                -470297.35
            ])  # Units: J/mol
        else:  # n
            A_k = np.array([
                86.19
            ])  # Units: J/mol

        summation = np.empty(len(A_k))
        for k in range(len(A_k)):
            summation[k] = A_k[k] * ((2 * x - 1) ** (k + 1)) - ((2 * x * k * (1 - x)) / (2 * x - 1) ** (1 - k))

        V = (1 / (n * F)) * np.sum(summation)

        return V

    def V_U(self, n_p):
        R = 8.314  # J/mol Ideal Gas Constant
        F = 96485.33289  # Faraday's Constant
        n = 1  # for Li-ion battery
        T = self.temperature + 273.15
        if n_p == 'p':  # p
            x_p = self.capacity
            U_0 = 4.03  # Units: Volts
            x_s = x_p
            V_INT = self.V_INT('p', x_p)
        else:  # n
            x_n = self.capacity
            U_0 = 0.01  # Units: Volts
            x_s = x_n
            V_INT = self.V_INT('p', x_n)
        V = U_0 + (R*T)/(n*F) * math.log((1 - x_s)/x_s) + V_INT
        return V

    def V_0(self, current):
        i_app = current
        R_0 = 0.085
        V = i_app * R_0
        return V

    def V_eta(self, n_p):
        R = 8.314  # J/mol Ideal Gas Constant
        F = 96485.33289  # Faraday's Constant
        T = self.temperature + 273.15
        alpha = 0.5
        if n_p == 'p':  # p
            S = 2 * 10**-4  # m^2
            k = 2 * 10**4  # A/m^2
            x_s = self.capacity
        else:  # n
            S = 2 * 10 ** -4  # m^2
            k = 2 * 10 ** 4  # A/m^2
            x_s = self.capacity
        J = self.current / S
        J_0 = k * ((1 - x_s)**alpha) * x_s**(1 - alpha)
        V = (R*T)/(F*alpha) * math.asinh(J/(2 * J_0))
        return V

    def V(self):
        V = self.V_U('p') - self.V_U('n') - self.V_0(self.current) - self.V_eta('p') - self.V_eta('n')
        return V

    def update_parameters_cc_charge(self):
        self.step()
        self.voltage = self.V()
        self.current = self.current  # Constant
        self.save_state()
        print(f'Voltage: {self.voltage}; Capacity: {self.capacity}')

    def update_parameters_cv_charge(self):
        self.step()
        self.voltage = self.get_constant_voltage()
        self.current = self.current + (0.28 * math.e ** (self.current*3.2))
        self.save_state()
        print(f'Voltage: {self.voltage}; Capacity: {self.capacity}')

    def update_parameters_cc_discharge(self):
        self.step()
        self.voltage = self.V()
        self.current = self.current  # Constant
        self.save_state()
        print(f'Voltage: {self.voltage}; Capacity: {self.capacity}')

    def run(self):
        self.save_cycle()
        self.save_state()
        for cycle in range(1, self.cycles+1):
            print(f'Simulating cycle {cycle}...')
            self.cc_charge()
            self.cv_charge()
            self.cc_discharge()
            print(f'Finished simulating cycle {cycle}.')
            self.save_cycle()

        self.plot()

    def cc_charge(self):
        print(f'Charging with constant current ({self.charging_current}A)')
        self.current = self.charging_current
        while self.voltage < self.battery_specs.charged_voltage:
            self.update_parameters_cc_charge()

    def cv_charge(self):
        print(f'Charging with constant voltage ({self.battery_specs.charged_voltage}V)')
        while self.current < self.cutoff_current:
            self.update_parameters_cv_charge()

    def cc_discharge(self):
        print(f'Discharging with constant current ({self.discharging_current}A)')
        self.current = self.discharging_current
        while self.voltage > self.battery_specs.discharged_voltage:
            self.update_parameters_cc_discharge()

    def plot(self):
        self.time_t = np.array(self.time_t)
        self.capacity_t = np.array(self.capacity_t)
        self.voltage_t = np.array(self.voltage_t)
        # plt.figure()
        # plt.plot(self.time_t, self.capacity_t)
        # plt.figure()
        # plt.plot(self.time_t, self.voltage_t)
        # plt.show()
        plt.figure()
        plt.title('One Cycle Capacity-Voltage Phase Space')
        plt.xlabel('Capacity (Ah)')
        plt.ylabel('Voltage (V)')
        plt.plot(self.capacity_t, self.voltage_t)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', help='Number of charge/discharge cycles to run the simulation for', type=int)
    parser.add_argument('--time_step_mins', help='The simulation time step, in minutes', type=float)
    parser.add_argument('--charging_current', help='The charging current, in amps', type=float)
    parser.add_argument('--discharging_current', help='The discharging current, in amps', type=float)
    kwargs = vars(parser.parse_args())
    for arg in [null_field for null_field in kwargs if kwargs[null_field] is None]:
        kwargs.pop(arg)
    Simulation = Simulate(**kwargs)
    Simulation.run()
