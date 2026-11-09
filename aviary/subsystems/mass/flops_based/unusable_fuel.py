import openmdao.api as om

from aviary.constants import FUEL_DENSITIES
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_thrust_factor,
)
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import FuelType
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportUnusableFuelMass(om.ExplicitComponent):
    """
    Calculates the mass of unusable fuel using the default method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    Assumptions
    -----------.
    All engines use fuel, single engine model on aircraft (both engine level thrust
    (scaled by num_engines) and propulsion-level count factor is being used)
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.NUM_TANKS)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Fuel.TYPE)

    def setup(self):
        num_fuel_types = len(self.options[Aircraft.Fuel.TYPE])

        add_aviary_input(
            self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, units='unitless', shape=num_fuel_types
        )
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/galUS', shape=num_fuel_types)
        add_aviary_input(self, Aircraft.Fuel.MAX_CAPACITY_MASS, units='lbm', shape=num_fuel_types)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        # add_aviary_output(
        #     self, Aircraft.Fuel.MAX_CAPACITY_VOLUME, units='galUS'
        # )  # not computed nor used
        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm', shape=num_fuel_types)

    def setup_partials(self):
        # self.declare_partials(
        #     Aircraft.Fuel.MAX_CAPACITY_VOLUME,
        #     [Aircraft.Fuel.MAX_CAPACITY_MASS, Aircraft.Fuel.DENSITY],
        # )
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        self.declare_partials(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
            rows=[jet_a_idx],
            cols=[jet_a_idx],
        )
        self.declare_partials(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            Aircraft.Fuel.DENSITY,
            rows=[jet_a_idx],
            cols=[jet_a_idx],
        )
        self.declare_partials(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            [
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                Aircraft.Wing.AREA,
                Aircraft.Fuel.MAX_CAPACITY_MASS,
            ],
            rows=[jet_a_idx],
            cols=[0],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)
        tank_count = self.options[Aircraft.Fuel.NUM_TANKS]

        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER][jet_a_idx]
        # Calculate fuel density ratio relative to Jet A: 6.7 lbm/galUS = 50.11948 lbm/ft**3
        density_ratio = inputs[Aircraft.Fuel.DENSITY][jet_a_idx] / wrapped_convert_units(
            FUEL_DENSITIES[FuelType.JET_A], 'lbm/galUS'
        )
        total_capacity = inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST][0]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)
        wing_area = inputs[Aircraft.Wing.AREA][0]

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS][jet_a_idx] = (
            (
                11.5 * num_eng_fact * thrust_factor**0.2
                + 0.07 * wing_area
                + 1.6 * tank_count * total_capacity**0.28
            )
            * density_ratio
        ) * scaler

    def compute_partials(self, inputs, J):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)
        tank_count = self.options[Aircraft.Fuel.NUM_TANKS]

        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER][jet_a_idx]
        density_ratio = inputs[Aircraft.Fuel.DENSITY][jet_a_idx] / wrapped_convert_units(
            FUEL_DENSITIES[FuelType.JET_A], 'lbm/galUS'
        )
        total_capacity = inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST][0]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)
        wing_area = inputs[Aircraft.Wing.AREA][0]

        term1 = thrust_factor**0.2
        term2 = total_capacity**0.28

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER] = (
            11.5 * num_eng_fact * term1 + 0.07 * wing_area + 1.6 * tank_count * term2
        ) * density_ratio

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = (
            2.3 * thrust_factor**-0.8 * density_ratio * scaler
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Wing.AREA] = 0.07 * density_ratio * scaler

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.MAX_CAPACITY_MASS] = (
            0.448 * tank_count * total_capacity**-0.72 * density_ratio * scaler
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.DENSITY] = (
            (11.5 * num_eng_fact * term1 + 0.07 * wing_area + 1.6 * tank_count * term2)
            / wrapped_convert_units(FUEL_DENSITIES[FuelType.JET_A], 'lbm/galUS')
        ) * scaler


class AltUnusableFuelMass(om.ExplicitComponent):
    """
    Calculates the mass of unusable fuel using the alternate method.

    The methodology is based on the FLOPS weight equations, modified to output mass instead of
    weight.
    """

    def initialize(self):
        # add_aviary_option(self, Aircraft.Fuel.NUM_TANKS)
        # add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Fuel.TYPE)

    def setup(self):
        num_fuel_types = len(self.options[Aircraft.Fuel.TYPE])

        add_aviary_input(
            self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, units='unitless', shape=num_fuel_types
        )

        add_aviary_input(self, Aircraft.Fuel.MAX_CAPACITY_MASS, units='lbm', shape=num_fuel_types)

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm', shape=num_fuel_types)

    def setup_partials(self):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        self.declare_partials(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            [Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, Aircraft.Fuel.MAX_CAPACITY_MASS],
            rows=[jet_a_idx],
            cols=[jet_a_idx],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        total_fuel_capacity = inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER][jet_a_idx]

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS][jet_a_idx] = 0.0084 * total_fuel_capacity * scaler

    def compute_partials(self, inputs, J):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        total_fuel_capacity = inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER][jet_a_idx]

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER] = (
            0.0084 * total_fuel_capacity
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.MAX_CAPACITY_MASS] = 0.0084 * scaler
