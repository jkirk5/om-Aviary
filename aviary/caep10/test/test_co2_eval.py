import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import aviary.api as av
from aviary.caep10.co2_evaluation import CO2EmissionsMetric


class TestCO2Emissions(unittest.TestCase):
    def test_postmission(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'emissions_metric',
            subsys=CO2EmissionsMetric(),
            promotes=['*'],
        )

        prob.model.set_input_defaults('inv_sar_avg', 0.662, units='kg/km')
        prob.model.set_input_defaults('reference_geometry_factor', 125.8824, units='unitless')
        prob.model.set_input_defaults(av.Aircraft.Design.GROSS_MASS, 160_000, units='lbm')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        # There are no truth values for these.
        expected_values = {
            ('CO2_emissions_factor', 'kg/km'): 0.20742753,
            ('CO2_emissions_factor_maximum', 'kg/km'): 0.7772986,
        }

        for (var_name, units), expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(
                    prob.get_val(var_name, units=units), expected_val, tolerance=1e-10
                )

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
