import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import aviary.api as av
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor


class TestReferenceGeometry(unittest.TestCase):
    def test_postmission(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'geometry_factor',
            subsys=ReferenceGeometryFactor(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(av.Aircraft.Fuselage.MAX_WIDTH, 3.5, units='m')
        prob.model.set_input_defaults(
            av.Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 118, units='ft'
        )

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        rgf = prob.get_val('reference_geometry_factor')  # nondimensional
        rgf_expected = 125.8824

        assert_near_equal(rgf, rgf_expected, tolerance=1e-10)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
