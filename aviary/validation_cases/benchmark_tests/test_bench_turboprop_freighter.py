'''
NOTES:
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
Large Turboprop Freighter data
'''

import unittest

import numpy as np
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse
from openmdao.core.problem import _clear_problem_names

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.enums import Verbosity
from aviary.validation_cases.benchmark_utils import \
    compare_against_expected_values
from aviary.models.large_turboprop_freighter.phase_info import phase_info
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    def setUp(self):
        self.phase_info = phase_info

    @require_pyoptsparse(optimizer="SNOPT")
    def bench_test_1(self):
        prob = AviaryProblem()

        prob.load_inputs('models/large_turboprop_freighter/large_turboprop_freighter.csv',
                         phase_info)

        options = prob.aviary_inputs

        turboshaft = options.get_val('engine_models')[0]
        turboprop = TurbopropModel('turboprop', options, shaft_power_model=turboshaft)
        options.set_val('engine_models', [turboprop])
        preprocess_propulsion(options)

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.add_driver("SNOPT", max_iter=50, use_coloring=True)

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_aviary_problem("dymos_solution.db", suppress_solver_print=True)

        # compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    test = ProblemPhaseTestCase()
    test.setUp()
    test.bench_test_1()
