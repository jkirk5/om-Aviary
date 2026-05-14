import numpy as np
import openmdao.api as om
from caep10 import calc_co2_metric, calc_max_permitted, calc_rgf, calc_sar
from openmdao.utils.assert_utils import assert_near_equal

from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.caep10.specific_air_range import inv_sar_avg, sar_calc
from aviary.core.aviary_problem import AviaryProblem
from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.utils.named_values import NamedValues
from aviary.validation_cases.validation_tests import get_flops_inputs, get_flops_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic

inputs = NamedValues(
    {
        Dynamic.Mission.ALTITUDE_RATE: (np.array([0, 0, 0]), 'ft/s'),
        Dynamic.Mission.VELOCITY_RATE: (np.array([0, 0, 0]), 'ft/s**2'),
        Dynamic.Atmosphere.MACH_RATE: (np.array([0, 0, 0]), '1/s'),
        Dynamic.Atmosphere.MACH: (np.array([0.82, 0.82, 0.82]), 'unitless'),
        Dynamic.Mission.ALTITUDE: (np.array([30000, 30000, 30000]), 'ft'),
    }
)

## OPENMDAO IMPLEMENTATION ##
av_prob = AviaryProblem()
av_prob.load_inputs('advanced_single_aisle_FLOPS')
av_prob.check_and_preprocess_inputs()
av_prob.build_model()
subsystems = av_prob.model.subsystems
aviary_inputs = av_prob.model.aviary_inputs

ode = EnergyStateODE(
    num_nodes=3,
    aviary_options=aviary_inputs,
    subsystems=subsystems,
)

# calculate masses for each evaluation point
mass_calc_high = om.ExecComp()
mass_calc_high.add_expr(
    'mass_high = 0.92 * togm',
    mass_high={'val': np.ones(1), 'units': 'kg'},
    togm={'val': 1.0, 'units': 'kg'},
)
mass_calc_low = om.ExecComp()
mass_calc_low.add_expr(
    'mass_low = (0.45 * togm) + (0.63 * power(togm, 0.924))',
    mass_low={'val': np.ones(1), 'units': 'kg'},
    togm={'val': 1.0, 'units': 'kg'},
)
mass_calc_mid = om.ExecComp()
mass_calc_mid.add_expr(
    'mass_mid = (mass_high + mass_low) / 2',
    mass_mid={'val': np.ones(1), 'units': 'kg'},
    mass_high={'val': np.ones(1), 'units': 'kg'},
    mass_low={'val': np.ones(1), 'units': 'kg'},
)
mass_calc = om.Group()
mass_calc.add_subsystem(
    'mass_calc_high',
    mass_calc_high,
    promotes_inputs=[('togm', Aircraft.Design.GROSS_MASS)],
    promotes_outputs=['*'],
)
mass_calc.add_subsystem(
    'mass_calc_low',
    mass_calc_low,
    promotes_inputs=[('togm', Aircraft.Design.GROSS_MASS)],
    promotes_outputs=['*'],
)
mass_calc.add_subsystem(
    'mass_calc_mid', mass_calc_mid, promotes_inputs=['*'], promotes_outputs=['*']
)


class SpecificAirRangeGroup(om.Group):
    def setup(self):
        # calculate SAR for each point
        self.add_subsystem('sar_calc', sar_calc, promotes_inputs=['*'], promotes_outputs=['*'])

        # calculate average SAR
        self.add_subsystem(
            'inverse_sar_average',
            inv_sar_avg,
            promotes_inputs=['sar_1', 'sar_2', 'sar_3'],
            promotes_outputs=['inv_sar_avg'],
        )


prob = om.Problem()

prob.model.add_subsystem(
    'mass_calc',
    mass_calc,
    promotes_inputs=[Aircraft.Design.GROSS_MASS],
    promotes_outputs=['mass_high', 'mass_mid', 'mass_low'],
)
mass_mux = om.MuxComp(vec_size=3)
mass_mux.add_var(Dynamic.Vehicle.MASS, val=1.0, units='lbm')
prob.model.add_subsystem(
    'mass_mux',
    mass_mux,
    promotes_inputs=[('mass_0', 'mass_high'), ('mass_1', 'mass_mid'), ('mass_2', 'mass_low')],
    promotes_outputs=['*'],
)
prob.model.add_subsystem('cruise_perf', ode, promotes=['*'])
prob.model.add_subsystem('sar_group', SpecificAirRangeGroup(), promotes=['*'])
for i in range(0, 3):
    prob.model.connect(Dynamic.Mission.VELOCITY, f'tas_{i + 1}', src_indices=om.slicer[i])
    prob.model.connect(
        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        f'w_f_{i + 1}',
        src_indices=om.slicer[i],
    )
prob.model.add_subsystem('rgf', ReferenceGeometryFactor(), promotes=['*'])
prob.model.add_subsystem('co2_eval', CO2EmissionsMetric(), promotes=['*'])

flops_inputs = get_flops_inputs('AdvancedSingleAisle')
flops_outputs = get_flops_outputs('AdvancedSingleAisle')

setup_model_options(prob, flops_inputs)
# setup_model_options(prob, get_flops_outputs('AdvancedSingleAisle'))
setup_model_options(prob, aviary_inputs)

prob.setup()
om.n2(prob, show_browser=False)
for input, (val, units) in inputs.items():
    if units == 'lbm':
        val = inputs.get_val(input, 'kg')
        units = 'kg'
    if units == 'knot':
        val = inputs.get_val(input, 'm/s')
        units = 'm/s'
    if units == 'lbm/h':
        val = inputs.get_val(input, 'kg/s')
        units = 'kg/s'
    prob.set_val(input, val, units)

for input, (val, units) in flops_inputs:
    try:
        prob.set_val(input, val, units)
    except KeyError:
        pass
for input, (val, units) in flops_outputs:
    try:
        prob.set_val(input, val, units)
    except KeyError:
        pass

prob.run_model()
om.n2(prob, show_browser=False)

# cruise data (inputs to original method)
tas_list = prob.get_val(Dynamic.Mission.VELOCITY, 'knot')
w_f_list = -1 * prob.get_val(Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, 'lbm/h')
mtom_kg = prob.get_val(Aircraft.Design.GROSS_MASS, 'kg')

# OM outputs
mass_high_om = prob.get_val('mass_high', 'kg')
mass_low_om = prob.get_val('mass_low', 'kg')
mass_mid_om = prob.get_val('mass_mid', 'kg')
sar_1_om = prob.get_val('sar_1', 'km/kg')
sar_2_om = prob.get_val('sar_2', 'km/kg')
sar_3_om = prob.get_val('sar_3', 'km/kg')
inv_sar_avg_om = prob.get_val('inv_sar_avg', 'kg/km')
rgf_om = prob.get_val('reference_geometry_factor')
co2_om = prob.get_val('CO2_emissions_factor')
co2_max_om = prob.get_val('CO2_emissions_factor_maximum')

## ORIGINAL IMPLEMENTATION ##
# mtom_kg = inputs.get_val(Aircraft.Design.GROSS_MASS, 'kg')
mass_high_py = 0.92 * mtom_kg  # kg
mass_low_py = (0.45 * mtom_kg) + (0.63 * mtom_kg**0.924)  # kg
mass_mid_py = (mass_high_py + mass_low_py) / 2  # kg

rgf_py = calc_rgf(
    prob.model.get_val(Aircraft.Fuselage.MAX_WIDTH, 'ft'),
    prob.model.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 'ft'),
)
# tas_list = np.array(
#     [
#         inputs.get_val('tas_1', 'knot'),
#         inputs.get_val('tas_2', 'knot'),
#         inputs.get_val('tas_3', 'knot'),
#     ]
# )
# w_f_list = np.array(
#     [
#         inputs.get_val('w_f_1', 'lbm/h'),
#         inputs.get_val('w_f_2', 'lbm/h'),
#         inputs.get_val('w_f_3', 'lbm/h'),
#     ]
# )

sar_inv_av_py = calc_sar(tas_list, w_f_list)

co2_py = calc_co2_metric(sar_inv_av_py, rgf_py)
co2_max_py = calc_max_permitted(prob.model.get_val(Aircraft.Design.GROSS_MASS, 'lbm'))

# assert_near_equal(mass_high_om, mass_high_py)
# assert_near_equal(mass_low_om, mass_low_py)
# assert_near_equal(mass_mid_om, mass_mid_py)
# assert_near_equal(inv_sar_avg_om, sar_inv_av_py)
# assert_near_equal(rgf_om, rgf_py)
# assert_near_equal(co2_om, co2_py)
# assert_near_equal(co2_max_om, co2_max_py)

print(f'mass_high: {mass_high_py}')
print(f'mass_mid: {mass_mid_py}')
print(f'mass_low: {mass_low_py}')
print(f'inv sar_avg: {sar_inv_av_py}')
print(f'rgf: {rgf_py}')
print(f'co2: {co2_py}')
print(f'cos_max: {co2_max_py}')

print(f'tas: {tas_list}')
print(f'w_f: {w_f_list}')
