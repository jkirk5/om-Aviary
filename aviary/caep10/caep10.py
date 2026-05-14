"""
Adapted from script provided by Dahlia Pham on 4/14/2026
"""

import numpy as np
import openmdao.api as om

# from aircraft_dictionary.helper import add_input_var
# from aircraft_dictionary.variables import Aircraft, Mission
from scipy.interpolate import interp1d

# from concepts_library.utils.flops_functions import parse_cruise

ft_to_m = 0.3048  # exactly
kts_to_kmh = 1.852  # exactly
lb_to_kg = lbh_to_kgh = 0.45359237  # exactly


def calc_rgf(fuselage_width_ft, fuselage_length_ft):
    """
    Calculate Reference Geometry Factor (RFG)
    """
    fuselage_width_m = fuselage_width_ft * ft_to_m  # m
    fuselage_length_m = fuselage_length_ft * ft_to_m  # m

    surface_area = fuselage_width_m * fuselage_length_m  # m^2
    rgf = surface_area / 1.0  # normalize by 1 m^2

    return rgf


def calc_sar(sar_tas_kts, sar_ff_lbm_hr):
    # calculate all three sars
    sar_inv = []
    for i in range(len(sar_tas_kts)):
        sar_tas = sar_tas_kts[i] * kts_to_kmh  # km/h
        sar_ff_kg_hr = sar_ff_lbm_hr[i] * lbh_to_kgh  # kg/h
        sar_inv.append(sar_ff_kg_hr / sar_tas)

    # average SAR
    sar_avg_inv = np.average(sar_inv)

    return sar_avg_inv


def calc_co2_metric(sar_avg_inv, rgf):

    co2_metric = sar_avg_inv / rgf**0.24

    return co2_metric


def calc_max_permitted(mtom_lb):
    mtom_kg = mtom_lb * lb_to_kg  # kg
    log_mtom = np.log10(mtom_kg)

    if mtom_kg <= 60000:
        max_emissions = 10 ** (-2.73780 + (0.681310 * log_mtom) - (0.0277861 * log_mtom**2))
        mtom_category = '=< 60,000 kg'
    elif mtom_kg <= 70395:
        max_emissions = 0.764
        mtom_category = '60,000 < MTOM <= 70,395 kg'
    else:
        max_emissions = 10 ** (-1.412742 + (-0.020517 * log_mtom) + (0.0593831 * log_mtom**2))
        mtom_category = '> 70,395 kg'

    print('max.permitted CO2 ICAO Index')
    print(f'MTOM = {mtom_kg} kg [{mtom_category}]')
    print(f'Maximum Permitted = {max_emissions} kg/km')

    return max_emissions


class caep10(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('flops_output', desc='Filename for FLOPS output file')

    def setup(self):
        add_input_var(self, Mission.Design.GROSS_MASS)
        add_input_var(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        add_input_var(self, Aircraft.Fuselage.MAX_WIDTH)
        self.add_output('caep10_co2', val=0.0, desc='CAEP10 CO2 emissions metric', units='kg/km')
        self.add_output(
            'max_permitted_caep10_co2',
            val=0.0,
            desc='Max permitted CAEP10 CO2 emissions metric',
            units='kg/km',
        )

    def compute(self, inputs, outputs):
        # inputs
        flops_output = self.options['flops_output']
        mtom_lb = inputs[Mission.Design.GROSS_MASS]  # lbm
        mtom_kg = mtom_lb * lb_to_kg  # kg
        fuselage_width_ft = inputs[Aircraft.Fuselage.MAX_WIDTH]  # ft
        fuselage_length_ft = inputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH]  # ft
        mtom_lb = inputs[Mission.Design.GROSS_MASS]  # lbm

        # calculate SAR masses
        sar1_mass = 0.92 * mtom_kg  # kg
        sar3_mass = (0.45 * mtom_kg) + (0.63 * mtom_kg**0.924)  # kg
        sar2_mass = (sar1_mass + sar3_mass) / 2  # kg

        # assign outputs in lbm
        sar1_mass_lbm = sar1_mass * (1 / lb_to_kg)  # lbm
        sar2_mass_lbm = sar2_mass * (1 / lb_to_kg)  # lbm
        sar3_mass_lbm = sar3_mass * (1 / lb_to_kg)  # lbm
        sar_masses_lbm = [sar1_mass_lbm, sar2_mass_lbm, sar3_mass_lbm]  # lbm

        # parse flops cruise output to get mass, velocity, fuel_flow
        m, v, ff = parse_cruise(flops_output, instance=1)

        # interpolate to get sar values
        sar_tas_kts = []  # knots
        sar_ff_lbm_hr = []  # lbm/h
        for i in range(len(sar_masses_lbm)):
            f_v = interp1d(m, v, kind='linear', fill_value='extrapolate')  # knots
            f_ff = interp1d(m, ff, kind='linear', fill_value='extrapolate')  # lbm/h

            sar_tas_kts.append(f_v(sar_masses_lbm[i]))  # knots
            sar_ff_lbm_hr.append(f_ff(sar_masses_lbm[i]))  # lbm/h

        sar_avg_inv = calc_sar(sar_tas_kts, sar_ff_lbm_hr)

        # calculate rfg
        rgf = calc_rgf(fuselage_width_ft, fuselage_length_ft)

        # calculate metric and max permitted
        outputs['caep10_co2'] = calc_co2_metric(sar_avg_inv, rgf)  # kg/km
        outputs['max_permitted_caep10_co2'] = calc_max_permitted(mtom_lb)  # kg/km


if __name__ == '__main__':
    model = om.Group()
    model.add_subsystem(
        'caep10',
        caep10(flops_output='concepts_library/cc2035/reference_model/cc2035.flopsout'),
        promotes=['*'],
    )

    prob = om.Problem(model)
    prob.setup(force_alloc_complex=True)

    prob.set_val(Mission.Design.GROSS_MASS, 170587.4, units='lbm')
    prob.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 85.50, units='ft')  # ft
    prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, units='ft')

    prob.run_model()

    print('caep10_co2', prob.get_val('caep10_co2'))  # kg/km
    print('max_permitted_caep10_co2', prob.get_val('max_permitted_caep10_co2'))  # kg/km
