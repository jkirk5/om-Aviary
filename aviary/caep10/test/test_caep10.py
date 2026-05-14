import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.caep10.caep10_group import CAEP10EmissionsGroup
from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.mass_points import mass_points_calc
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.caep10.specific_air_range import SpecificAirRangeGroup
from aviary.core.aviary_problem import AviaryProblem
from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.named_values import NamedValues
from aviary.validation_cases.validation_tests import get_flops_inputs, get_flops_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic


class TestCAEP10Group(unittest.TestCase):
    def test_emissions_group_HE(self):
        inputs = NamedValues(
            {
                Dynamic.Mission.ALTITUDE_RATE: (np.array([0, 0, 0]), 'm/s'),
                Dynamic.Mission.VELOCITY_RATE: (np.array([0, 0, 0]), 'm/s**2'),
                Dynamic.Atmosphere.MACH_RATE: (np.array([0, 0, 0]), '1/s'),
                Dynamic.Atmosphere.MACH: (np.array([0.82, 0.82, 0.82]), 'unitless'),
                Dynamic.Mission.ALTITUDE: (np.array([9144, 9144, 9144]), 'm'),  # 9144m = 30000ft
            }
        )

        # dummy AviaryProblem just to shortcut building subsystems with correct options
        av_prob = AviaryProblem()
        av_prob.load_inputs('advanced_single_aisle_FLOPS')
        av_prob.check_and_preprocess_inputs()
        av_prob.build_model()
        subsystems = av_prob.model.subsystems
        aviary_inputs = av_prob.model.aviary_inputs

        prob = om.Problem()
        prob.model.add_subsystem(
            'emissions',
            CAEP10EmissionsGroup(subsystems=subsystems, aviary_options=aviary_inputs),
            promotes=['*'],
        )

        flops_inputs = get_flops_inputs('AdvancedSingleAisle')
        flops_outputs = get_flops_outputs('AdvancedSingleAisle')

        # setup_model_options(prob, flops_inputs)
        # setup_model_options(prob, get_flops_outputs('AdvancedSingleAisle'))
        setup_model_options(prob, aviary_inputs)

        prob.setup()

        prob.set_val(Dynamic.Atmosphere.MACH, [0.82, 0.82, 0.82])
        prob.set_val(Dynamic.Mission.ALTITUDE, [9144, 9144, 9144], units='m')
        prob.set_val(Dynamic.Mission.ALTITUDE_RATE, [0, 0, 0], 'm/s')
        # prob.set_val(Dynamic.Mission.VELOCITY_RATE, [0, 0, 0], 'm/s**2')
        prob.set_val(Dynamic.Atmosphere.MACH_RATE, [0, 0, 0], '1/s')

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

        try:
            prob.run_model()
        except:
            pass
        # prob.model.emissions.cruise_perf.problem.model.list_vars(print_arrays=True, units=True)
        # prob.model.emissions.cruise_perf.problem.model.list_options(
        #     include_default=False, include_solvers=False
        # )
        om.n2(prob, outfile='problem.html', show_browser=False)
        om.n2(
            prob.model.emissions.cruise_perf.problem, outfile='subproblem.html', show_browser=False
        )

        expected_values = {
            Dynamic.Vehicle.MASS: ([54138.64, 48354.85, 42571.06], 'kg'),
            'inv_sar_avg': (2.036315582214561, 'kg/km'),
            'reference_geometry_factor': (111.41397072, 'unitless'),
            'CO2_emissions_factor': (0.65702148, 'kg/km'),
            'CO2_emissions_factor_maximum': (0.75808449, 'kg/km'),
            Dynamic.Mission.VELOCITY: ([483.28790309, 483.28790309, 483.28790309], 'knot'),
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: (
                [-4147.75386454, -3997.43969914, -3909.25937424],
                'lbm/h',
            ),
        }
        # print(prob.model.emissions.cruise_perf.problem.get_val(Dynamic.Atmosphere.MACH))
        # print(prob.model.emissions.cruise_perf.problem.get_val(Dynamic.Mission.ALTITUDE))
        # print(prob.model.emissions.cruise_perf.problem.get_val(Dynamic.Mission.VELOCITY_RATE))
        for var_name, (expected_val, units) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units), expected_val, 1e-6)


# mass_high: [54138.64432721]
# mass_mid: [48354.85093625]
# mass_low: [42571.05754528]
# inv sar_avg: 2.036315582214561
# rgf: [111.41397072]
# co2: [0.65702148]
# cos_max: [0.75808449]
# tas: [483.28790309 483.28790309 483.28790309]
# w_f: [4147.75386454 3997.43969914 3909.25937424]

if __name__ == '__main__':
    unittest.main()
