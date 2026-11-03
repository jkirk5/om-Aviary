"""Test preprocessing as part of the level 2 interface."""

import unittest
import warnings
from copy import deepcopy

import numpy as np
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.phase_info import phase_info
from aviary.subsystems.energy.fuel_tank_model import FuelTankModel
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_fuel, preprocess_propulsion
from aviary.variable_info.enums import FuelType
from aviary.variable_info.variables import Aircraft, Settings


class MethaneEngine(EngineModel):
    """An EngineModel that burns liquid methane, for exercising multi-fuel branches."""

    fuel_type = FuelType.METHANE_LIQUID


class MethaneFuelTank(FuelTankModel):
    """A FuelTankModel that stores liquid methane, for exercising multi-fuel branches."""

    fuel_type = FuelType.METHANE_LIQUID


def _make_engine(name: str = 'engine', **options) -> EngineModel:
    """Build a bare EngineModel with the given Aircraft.Engine options set."""
    engine = EngineModel(name=name)
    for key, val in options.items():
        engine.set_val(key, val, 'unitless')
    return engine


def _make_aviary_options(verbosity: int = 0, **options) -> AviaryValues:
    """Build an AviaryValues with Settings.VERBOSITY and any extra unitless options set."""
    aviary_options = AviaryValues()
    aviary_options.set_val(Settings.VERBOSITY, verbosity)
    for key, val in options.items():
        aviary_options.set_val(key, val)
    return aviary_options


def _make_aviary_options_with_units(verbosity: int = 0, **options) -> AviaryValues:
    """Build an AviaryValues; each kwarg value is a (val, units) tuple."""
    aviary_options = AviaryValues()
    aviary_options.set_val(Settings.VERBOSITY, verbosity)
    for key, (val, units) in options.items():
        aviary_options.set_val(key, val, units)
    return aviary_options


class TestPrePreprocessing(unittest.TestCase):
    @use_tempdirs
    def test_crew_preprocessing(self):
        # Test that flight-crew preprocesses correctly.
        prob = AviaryProblem()
        local_phase_info = deepcopy(phase_info)

        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()
        aviary_inputs = prob.aviary_inputs

        num_flight_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        self.assertEqual(num_flight_crew, 3)

        num_flight_attendants = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS)
        self.assertEqual(num_flight_attendants, 4)

        num_galley_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)
        self.assertEqual(num_galley_crew, 1)

    @use_tempdirs
    def test_missing_passengers_warnings(self):
        # Test that verifies Aviary's behavior when the number of passengers in each clase are
        # not consistent with given totals.
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.delete(Aircraft.CrewPayload.NUM_ECONOMY_CLASS)

        msg = (
            'Sum of all passenger classes (36) does not equal total number of '
            'passengers provided for current mission (154). Setting '
            'Aircraft.CrewPayload.NUM_PASSENGERS with the sum of passenger classes for flown '
            'mission (36).'
        )

        with assert_warning(UserWarning, msg):
            prob.check_and_preprocess_inputs()

        num_pass = prob.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
        self.assertEqual(num_pass, 36)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.delete(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 154)

        msg = (
            'Sum of all passenger classes (138) does not equal total number of '
            'passengers provided for aircraft design (154). Overriding '
            'Aircraft.CrewPayload.Design.NUM_PASSENGERS with the sum of '
            'passenger classes for design (138).'
        )

        with assert_warning(UserWarning, msg):
            prob.check_and_preprocess_inputs()

        num_pass = prob.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        self.assertEqual(num_pass, 138)

    @use_tempdirs
    def test_simple_cabin_layout(self):
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, False)

        prob.check_and_preprocess_inputs()


class PreprocessPropulsionTestCase(unittest.TestCase):
    """Tests for aviary.utils.preprocessors.preprocess_propulsion."""

    def test_vectorize_variable_priority_order(self):
        # Priority order per engine model is: 1. the engine model's own value,
        # 2. aviary_options at that engine's index, 3. the metadata default.
        # engine1's own value must win even though aviary_options also has a value at
        # index 0, proving the engine model takes priority over aviary_options.
        engine1 = _make_engine(
            name='engine1', **{Aircraft.Engine.NUM_ENGINES: 2, Aircraft.Engine.SCALE_FACTOR: 2.0}
        )
        engine2 = _make_engine(name='engine2', **{Aircraft.Engine.NUM_ENGINES: 2})
        engine3 = _make_engine(name='engine3', **{Aircraft.Engine.NUM_ENGINES: 2})
        aviary_options = _make_aviary_options(**{Aircraft.Engine.SCALE_FACTOR: [1.5, 1.6]})

        preprocess_propulsion(aviary_options, [engine1, engine2, engine3])

        scale_factors = aviary_options.get_val(Aircraft.Engine.SCALE_FACTOR)
        with self.subTest(engine='engine1 (own value overrides aviary_options)'):
            self.assertEqual(scale_factors[0], 2.0)
        with self.subTest(engine='engine2 (falls back to aviary_options)'):
            self.assertEqual(scale_factors[1], 1.6)
        with self.subTest(engine='engine3 (no value anywhere, uses metadata default)'):
            self.assertEqual(scale_factors[2], 1.0)

    def test_vectorize_multidimensional_variable(self):
        # Each engine model's own multidimensional values are concatenated in order,
        # not reduced to one value per engine model.
        engine1 = _make_engine(
            name='engine1',
            **{
                Aircraft.Engine.NUM_ENGINES: 4,
                Aircraft.Engine.NUM_WING_ENGINES: 4,
                Aircraft.Engine.WING_LOCATIONS: [0.3, 0.6],
            },
        )
        engine2 = _make_engine(
            name='engine2',
            **{
                Aircraft.Engine.NUM_ENGINES: 2,
                Aircraft.Engine.NUM_WING_ENGINES: 2,
                Aircraft.Engine.WING_LOCATIONS: [0.5],
            },
        )
        aviary_options = _make_aviary_options()

        preprocess_propulsion(aviary_options, [engine1, engine2])

        np.testing.assert_array_equal(
            aviary_options.get_val(Aircraft.Engine.WING_LOCATIONS), np.array([0.3, 0.6, 0.5])
        )

    def test_single_engine_defaults_to_fuselage_mount(self):
        # A single engine with no mount location specified is assumed fuselage-mounted.
        engine = _make_engine(**{Aircraft.Engine.NUM_ENGINES: 1})
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Mount location for single engine of type <engine> not specified.'
            'Assuming it is fuselage-mounted.'
        )
        with self.subTest(check='warning'), assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

        with self.subTest(var='NUM_FUSELAGE_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)[0], 1)
        with self.subTest(var='NUM_WING_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)[0], 0)

    def test_multiple_engines_default_to_wing_mount(self):
        # Multiple engines with no mount location specified are assumed wing-mounted.
        engine = _make_engine(**{Aircraft.Engine.NUM_ENGINES: 4})
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Mount location for engines of type <engine> not specified. Wing-'
            'mounted engines are assumed.'
        )
        with self.subTest(check='warning'), assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

        with self.subTest(var='NUM_WING_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)[0], 4)
        with self.subTest(var='NUM_FUSELAGE_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)[0], 0)

    def test_mount_counts_exceed_num_engines_overwrites_total(self):
        # If fuselage + wing mounted engines exceeds num_engines, num_engines is
        # overwritten with the sum.
        engine = _make_engine(
            **{
                Aircraft.Engine.NUM_ENGINES: 2,
                Aircraft.Engine.NUM_WING_ENGINES: 2,
                Aircraft.Engine.NUM_FUSELAGE_ENGINES: 2,
            }
        )
        aviary_options = _make_aviary_options(verbosity=1)

        msg = (
            'Sum of user-specified fuselage-mounted and wing-mounted engines do '
            'not match total number of engines for EngineModel <engine>. '
            'Overwriting total number of engines with the sum of wing and '
            'fuselage mounted engines.'
        )
        with assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

        self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)[0], 4)

    def test_mount_counts_below_num_engines_unspecified_assumed_wing(self):
        # If more than one engine's mount is unspecified, the remainder is assumed to be
        # wing-mounted.
        engine = _make_engine(
            **{Aircraft.Engine.NUM_ENGINES: 4, Aircraft.Engine.NUM_FUSELAGE_ENGINES: 1}
        )
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Mount location was not defined for all engines of EngineModel '
            '<engine> - unspecified engines are assumed wing-mounted.'
        )
        with self.subTest(check='warning'), assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

        with self.subTest(var='NUM_FUSELAGE_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)[0], 1)
        with self.subTest(var='NUM_WING_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)[0], 3)

    def test_mount_counts_below_num_engines_single_unspecified_assumed_fuselage(self):
        # If exactly one engine's mount is unspecified and wing engines are already
        # present, the remaining engine is assumed to be fuselage-mounted.
        engine = _make_engine(
            **{Aircraft.Engine.NUM_ENGINES: 3, Aircraft.Engine.NUM_WING_ENGINES: 2}
        )
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Mount location was not defined for all engine of EngineModel '
            '<engine> - unspecified engine is assumed fuselage-mounted.'
        )
        with self.subTest(check='warning'), assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

        with self.subTest(var='NUM_FUSELAGE_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)[0], 1)
        with self.subTest(var='NUM_WING_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)[0], 2)

    def test_odd_wing_engines_warns(self):
        # An odd number of wing-mounted engines triggers a warning about asymmetric
        # wing mass/geometry assumptions.
        engine = _make_engine(
            **{Aircraft.Engine.NUM_ENGINES: 3, Aircraft.Engine.NUM_WING_ENGINES: 3}
        )
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Odd number of wing engines are specified for EngineModel '
            '<engine> - this may cause issues with some mass and geometry '
            'components that assume symmetric wing engine distribution.'
        )
        with assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])

    def test_wing_engine_location_count_mismatch_raises(self):
        # Delegates to _validate_wing_engine_locations; a mismatched location count for
        # an engine model's own WING_LOCATIONS raises.
        engine = _make_engine(
            **{
                Aircraft.Engine.NUM_ENGINES: 4,
                Aircraft.Engine.NUM_WING_ENGINES: 4,
                Aircraft.Engine.WING_LOCATIONS: [0.3],
            }
        )
        aviary_options = _make_aviary_options()

        with self.assertRaisesRegex(ValueError, r'has 1 value\(s\).*require 2 location'):
            preprocess_propulsion(aviary_options, [engine])

    def test_propulsion_totals_are_summed_across_engine_models(self):
        # Aircraft.Propulsion.TOTAL_NUM_* variables are the sum across all engine models.
        engine1 = _make_engine(
            name='engine1', **{Aircraft.Engine.NUM_ENGINES: 4, Aircraft.Engine.NUM_WING_ENGINES: 4}
        )
        engine2 = _make_engine(
            name='engine2',
            **{Aircraft.Engine.NUM_ENGINES: 1, Aircraft.Engine.NUM_FUSELAGE_ENGINES: 1},
        )
        aviary_options = _make_aviary_options()

        preprocess_propulsion(aviary_options, [engine1, engine2])

        with self.subTest(var='TOTAL_NUM_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES), 5)
        with self.subTest(var='TOTAL_NUM_WING_ENGINES'):
            self.assertEqual(aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES), 4)
        with self.subTest(var='TOTAL_NUM_FUSELAGE_ENGINES'):
            self.assertEqual(
                aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES), 1
            )

    def test_verbosity_as_int_is_accepted(self):
        # verbosity may be passed as a plain int for backwards compatibility.
        engine = _make_engine(
            **{Aircraft.Engine.NUM_ENGINES: 2, Aircraft.Engine.NUM_WING_ENGINES: 2}
        )
        aviary_options = _make_aviary_options()

        preprocess_propulsion(aviary_options, [engine], verbosity=0)

        self.assertEqual(aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)[0], 2)

    def test_verbosity_read_from_aviary_options_when_not_passed(self):
        # If verbosity is not passed directly, it is read from aviary_options's
        # Settings.VERBOSITY.
        engine = _make_engine(**{Aircraft.Engine.NUM_ENGINES: 1})
        aviary_options = _make_aviary_options(verbosity=2)

        msg = (
            'Mount location for single engine of type <engine> not specified.'
            'Assuming it is fuselage-mounted.'
        )
        with assert_warning(UserWarning, msg):
            preprocess_propulsion(aviary_options, [engine])


class PreprocessFuelTestCase(unittest.TestCase):
    """Tests for aviary.utils.preprocessors.preprocess_fuel."""

    def test_no_fuel_tank_models_or_engine_models(self):
        # With nothing supplied, preprocess_fuel defaults to empty lists for both and
        # does not raise.
        aviary_options = _make_aviary_options()

        preprocess_fuel(aviary_options)

    def test_no_tanks_leaves_fuel_variables_untouched(self):
        # With no fuel tank models, the vectorization loop is skipped entirely and
        # Aircraft.Fuel.* variables are left as whatever aviary_options already had.
        aviary_options = _make_aviary_options()

        preprocess_fuel(aviary_options)

        with self.assertRaises(KeyError):
            aviary_options.get_val(Aircraft.Fuel.DENSITY)

    def test_density_vectorization(self):
        # Priority order per tank is: 1. aviary_options at that tank's index,
        # 2. the tank's own fuel_type-specific default from FUEL_DENSITIES (not the
        # metadata default, which is always Jet-A).
        # tank1 and tank2 get their value from aviary_options (proving aviary_options
        # overrides the default, and that it is vectorized per tank, not broadcast).
        # tank3 has no aviary_options value at its index, so it falls back to its own
        # fuel-type-specific default (methane, not Jet-A).
        tank1 = FuelTankModel(name='tank1')
        tank2 = FuelTankModel(name='tank2')
        tank3 = MethaneFuelTank(name='tank3')
        aviary_options = _make_aviary_options_with_units(
            **{Aircraft.Fuel.DENSITY: ([10.0, 20.0], 'lbm/galUS')}
        )

        preprocess_fuel(aviary_options, fuel_tank_models=[tank1, tank2, tank3])

        density = aviary_options.get_val(Aircraft.Fuel.DENSITY, 'lbm/galUS')
        with self.subTest(tank='tank1 (from aviary_options)'):
            self.assertAlmostEqual(density[0], 10.0)
        with self.subTest(tank='tank2 (from aviary_options)'):
            self.assertAlmostEqual(density[1], 20.0)
        with self.subTest(tank='tank3 (missing index, uses fuel-type default)'):
            self.assertAlmostEqual(density[2], 3.75)

    def test_non_density_variable_vectorized_from_metadata_default(self):
        # Variables other than DENSITY use the plain metadata default (not
        # FUEL_DENSITIES) when not supplied by aviary_options.
        tank = FuelTankModel()
        aviary_options = _make_aviary_options()

        preprocess_fuel(aviary_options, fuel_tank_models=[tank])

        unusable_mass = aviary_options.get_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 'lbm')
        self.assertEqual(unusable_mass[0], 0.0)

    def test_non_density_variable_from_aviary_options(self):
        # Non-density multivalue variables are also vectorized from aviary_options when
        # provided.
        tank1 = FuelTankModel(name='tank1')
        tank2 = FuelTankModel(name='tank2')
        aviary_options = _make_aviary_options_with_units(
            **{Aircraft.Fuel.MAX_CAPACITY_MASS: ([1000.0, 2000.0], 'lbm')}
        )

        preprocess_fuel(aviary_options, fuel_tank_models=[tank1, tank2])

        capacity = aviary_options.get_val(Aircraft.Fuel.MAX_CAPACITY_MASS, 'lbm')
        np.testing.assert_array_equal(capacity, np.array([1000.0, 2000.0]))

    def test_engine_fuel_type_without_matching_tank_warns(self):
        # If an engine model's fuel type has no corresponding tank, a warning is raised.
        engine = MethaneEngine(name='methane_engine')
        tank = FuelTankModel()
        aviary_options = _make_aviary_options(verbosity=1)

        msg = (
            'The following fuel types are used by engines but do not have a tank that stores '
            "that fuel: ['liquid methane']"
        )
        with assert_warning(UserWarning, msg):
            preprocess_fuel(aviary_options, fuel_tank_models=[tank], engine_models=[engine])

    def test_engine_fuel_type_with_matching_tank_does_not_warn(self):
        # If every engine's fuel type has a matching tank, no warning is raised.
        engine = MethaneEngine(name='methane_engine')
        tank = MethaneFuelTank()
        aviary_options = _make_aviary_options()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            preprocess_fuel(aviary_options, fuel_tank_models=[tank], engine_models=[engine])

        self.assertEqual(len(caught), 0)

    def test_verbosity_as_int_is_accepted(self):
        # verbosity may be passed as a plain int for backwards compatibility.
        engine = MethaneEngine(name='methane_engine')
        tank = FuelTankModel()
        aviary_options = _make_aviary_options()

        msg = (
            'The following fuel types are used by engines but do not have a tank that stores '
            "that fuel: ['liquid methane']"
        )
        with assert_warning(UserWarning, msg):
            preprocess_fuel(
                aviary_options, fuel_tank_models=[tank], engine_models=[engine], verbosity=1
            )

    def test_quiet_verbosity_suppresses_missing_tank_warning(self):
        # At QUIET verbosity, the missing-tank warning is suppressed.
        engine = MethaneEngine(name='methane_engine')
        tank = FuelTankModel()
        aviary_options = _make_aviary_options(verbosity=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            preprocess_fuel(aviary_options, fuel_tank_models=[tank], engine_models=[engine])

        self.assertEqual(len(caught), 0)


if __name__ == '__main__':
    unittest.main()
