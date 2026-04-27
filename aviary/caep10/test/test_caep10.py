import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import aviary.api as av
from aviary.caep10.caep10_builder import CAEP10Builder
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilder


class TestReferenceGeometry(unittest.TestCase):
    def test_postmission(self):
        prob = self.prob
        prob.model.add_subsystem(
            'CAEP10_emissions',
            subsys=CAEP10Builder().build_post_mission(
                num_nodes=4, aviary_inputs={}, user_options={}, subsystem_options={}
            ),
            promotes=['*'],
        )

        efficiency = 0.95
        prob.model.set_input_defaults(av.Aircraft.Battery.ENERGY_CAPACITY, 10_000, units='kJ')
        prob.model.set_input_defaults(av.Aircraft.Battery.EFFICIENCY, efficiency, units='unitless')
        prob.model.set_input_defaults(
            av.Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED,
            [0, 2_000, 5_000, 9_500],
            units='kJ',
        )

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        soc_expected = np.array([1.0, 0.7894736842105263, 0.4736842105263159, 0.0])
        soc = prob.get_val(av.Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE, 'unitless')

        assert_near_equal(soc, soc_expected, tolerance=1e-10)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)


class TestCAEP10Builder(av.TestSubsystemBuilder):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CAEP10Builder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
