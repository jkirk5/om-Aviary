import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft


class CO2EmissionsMetric(om.ExplicitComponent):
    def setup(self):
        self.add_input('inv_sar_avg', units='kg/km')
        self.add_input('reference_geometry_factor', units='unitless')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='kg')

        self.add_output('CO2_emissions_factor', units='kg/km')
        self.add_output('CO2_emissions_factor_maximum', units='kg/km')

    def setup_partials(self):
        self.declare_partials('CO2_emissions_factor', ['inv_sar_avg', 'reference_geometry_factor'])
        self.declare_partials('CO2_emissions_factor_maximum', [Aircraft.Design.GROSS_MASS])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inv_sar_avg = inputs['inv_sar_avg']
        rgf = inputs['reference_geometry_factor']
        gross_mass = inputs[Aircraft.Design.GROSS_MASS]

        co2_factor = inv_sar_avg / rgf**0.24

        co2_factor_max = 0.764  # 60,000 kg < gross mass <= 70,395 kg
        if gross_mass <= 60_000:
            exp = -2.7378 + 0.68131 * np.log10(gross_mass) - 0.0277861 * (np.log10(gross_mass) ** 2)
            co2_factor_max = 10**exp
        elif gross_mass > 70_395:
            exp = (
                -1.412742
                - 0.020517 * np.log10(gross_mass)
                + 0.0593831 * (np.log10(gross_mass) ** 2)
            )
            co2_factor_max = 10**exp

        outputs['CO2_emissions_factor'] = co2_factor
        outputs['CO2_emissions_factor_maximum'] = co2_factor_max

    def compute_partials(self, inputs, J, discrete_inputs=None):
        inv_sar_avg = inputs['inv_sar_avg']
        rgf = inputs['reference_geometry_factor']
        gross_mass = inputs[Aircraft.Design.GROSS_MASS]

        if gross_mass <= 60_000:
            log10_m = np.log10(gross_mass)
            exp = -2.7378 + 0.68131 * log10_m - 0.0277861 * log10_m**2
            co2_factor_max = 10.0**exp
            dexp_dm_times_m = 0.68131 - 2.0 * 0.0277861 * log10_m
            dco2_max_dgross_mass = co2_factor_max * dexp_dm_times_m / gross_mass
        elif gross_mass > 70_395:
            log10_m = np.log10(gross_mass)
            exp = -1.412742 - 0.020517 * log10_m + 0.0593831 * log10_m**2
            co2_factor_max = 10.0**exp
            dexp_dm_times_m = -0.020517 + 2.0 * 0.0593831 * log10_m
            dco2_max_dgross_mass = co2_factor_max * dexp_dm_times_m / gross_mass
        else:
            # Constant region: 60,000 < gross_mass <= 70,395
            dco2_max_dgross_mass = 0.0

        J['CO2_emissions_factor', 'inv_sar_avg'] = 1.0 / rgf**0.24
        J['CO2_emissions_factor', 'reference_geometry_factor'] = -0.24 * inv_sar_avg / rgf**1.24
        J['CO2_emissions_factor_maximum', Aircraft.Design.GROSS_MASS] = dco2_max_dgross_mass
