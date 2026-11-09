import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass,
    TransportUnusableFuelMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.enums import FuelType
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TransportUnusableFuelMassTest(unittest.TestCase):
    """Tests transport/GA unusable fuel mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'unusable_fuel',
            TransportUnusableFuelMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)
        prob.model_options['*'][Aircraft.Fuel.TYPE] = [FuelType.JET_A]

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.MAX_CAPACITY_MASS,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                Aircraft.Wing.AREA,
            ],
            output_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.TRANSPORT_and_BWB,
            tol=5e-4,
            excludes=['size_prop.*'],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_multi_fuel_type(self):
        prob = om.Problem()
        fuel_types = [FuelType.JET_A, FuelType.HYDROGEN_GASEOUS]

        prob.model.add_subsystem(
            'unusable_fuel',
            TransportUnusableFuelMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = {
            Aircraft.Fuel.TYPE: fuel_types,
            Aircraft.Fuel.NUM_TANKS: 7,
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: 2,
        }

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, [1.0, 1.0], units='unitless')
        prob.set_val(Aircraft.Fuel.DENSITY, [6.7, 2.0], units='lbm/galUS')
        prob.set_val(Aircraft.Fuel.MAX_CAPACITY_MASS, [20000.0, 500.0], units='lbm')
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 50000.0, units='lbf')
        prob.set_val(Aircraft.Wing.AREA, 1000.0, units='ft**2')

        prob.run_model()

        with self.subTest(var='jet_a_idx'):
            unusable_fuel_mass = prob.get_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')
            assert_near_equal(unusable_fuel_mass[0], 423.57691104, 1e-6)

        with self.subTest(var='other_fuel_idx'):
            assert_near_equal(unusable_fuel_mass[1], 0.0, 1e-6)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


@use_tempdirs
class AltUnusableFuelMassTest(unittest.TestCase):
    """Tests alternate unusable fuel mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'unusable_fuel', AltUnusableFuelMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.model_options['*'] = {Aircraft.Fuel.TYPE: [FuelType.JET_A]}

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, Aircraft.Fuel.MAX_CAPACITY_MASS],
            output_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_multi_fuel_type(self):
        prob = om.Problem()
        fuel_types = [FuelType.JET_A, FuelType.HYDROGEN_GASEOUS]

        prob.model.add_subsystem(
            'unusable_fuel', AltUnusableFuelMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.model_options['*'] = {
            Aircraft.Fuel.TYPE: fuel_types,
        }

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, [1.0, 1.0], units='unitless')
        prob.set_val(Aircraft.Fuel.MAX_CAPACITY_MASS, [20000.0, 500.0], units='lbm')

        prob.run_model()

        with self.subTest(var='jet_a_idx'):
            unusable_fuel_mass = prob.get_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')
            assert_near_equal(unusable_fuel_mass[0], 168.0, 1e-6)

        with self.subTest(var='other_fuel_idx'):
            assert_near_equal(unusable_fuel_mass[1], 0.0, 1e-6)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
