import openmdao.api as om

from aviary.variable_info.enums import FuelType
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class FuelCapacityGroup(om.Group):
    """
    Compute the maximum fuel that can be carried.

    These FLOPS equations model the combined capacity of tanks distributed around the
    aircraft holding Jet-A fuel; they do not apply to other fuel types, which supply
    their own capacity via their own fuel tank subsystem.
    """

    def setup(self):
        self.add_subsystem(
            'wing_fuel_mass_capacity',
            WingFuelCapacity(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'fuselage_fuel_mass_capacity',
            FuselageFuelCapacity(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'auxiliary_fuel_mass_capacity',
            AuxFuelCapacity(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'total_fuel_capacity',
            TotalFuelCapacity(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class FuselageFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the fuselage."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.TYPE)

    def setup(self):
        num_fuel_types = len(self.options[Aircraft.Fuel.TYPE])

        add_aviary_input(self, Aircraft.Fuel.MAX_CAPACITY_MASS, units='lbm', shape=num_fuel_types)
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_output(self, Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY, units='lbm')

    def setup_partials(self):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        self.declare_partials(
            Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY,
            Aircraft.Fuel.MAX_CAPACITY_MASS,
            rows=[0],
            cols=[jet_a_idx],
            val=1.0,
        )
        self.declare_partials(
            Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY,
            Aircraft.Fuel.WING_FUEL_MASS_CAPACITY,
            val=-1.0,
        )

    def compute(self, inputs, outputs):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        outputs[Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY] = (
            inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
            - inputs[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY]
        )


class AuxFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the auxiliary tanks."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.TYPE)

    def setup(self):
        num_fuel_types = len(self.options[Aircraft.Fuel.TYPE])

        add_aviary_input(self, Aircraft.Fuel.MAX_CAPACITY_MASS, units='lbm', shape=num_fuel_types)
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_output(self, Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY, units='lbm')

    def setup_partials(self):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY,
            Aircraft.Fuel.MAX_CAPACITY_MASS,
            rows=[0],
            cols=[jet_a_idx],
            val=1.0,
        )
        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY,
            Aircraft.Fuel.WING_FUEL_MASS_CAPACITY,
            val=-1.0,
        )
        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY,
            Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY,
            val=-1.0,
        )

    def compute(self, inputs, outputs):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        outputs[Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY] = (
            inputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx]
            - inputs[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY]
            - inputs[Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY]
        )


class TotalFuelCapacity(om.ExplicitComponent):
    """Compute the total Jet-A fuel that can be carried in all tanks."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.TYPE)

    def setup(self):
        num_fuel_types = len(self.options[Aircraft.Fuel.TYPE])

        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY, units='lbm')
        add_aviary_output(self, Aircraft.Fuel.MAX_CAPACITY_MASS, units='lbm', shape=num_fuel_types)

    def setup_partials(self):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        self.declare_partials(
            Aircraft.Fuel.MAX_CAPACITY_MASS,
            Aircraft.Fuel.WING_FUEL_MASS_CAPACITY,
            rows=[jet_a_idx],
            cols=[0],
            val=1.0,
        )
        self.declare_partials(
            Aircraft.Fuel.MAX_CAPACITY_MASS,
            Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY,
            rows=[jet_a_idx],
            cols=[0],
            val=1.0,
        )
        self.declare_partials(
            Aircraft.Fuel.MAX_CAPACITY_MASS,
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY,
            rows=[jet_a_idx],
            cols=[0],
            val=1.0,
        )

    def compute(self, inputs, outputs):
        fuel_types = self.options[Aircraft.Fuel.TYPE]
        jet_a_idx = fuel_types.index(FuelType.JET_A)

        outputs[Aircraft.Fuel.MAX_CAPACITY_MASS][jet_a_idx] = (
            inputs[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY]
            + inputs[Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY]
            + inputs[Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY]
        )


class WingFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the wing's enclosed space."""

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/ft**3')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        wing_area = inputs[Aircraft.Wing.AREA]

        fuel_density = inputs[Aircraft.Fuel.DENSITY]
        volume_fraction = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        span = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        # calculate volume of wing assuming truncated rectangular based pyramid:
        volume_of_wing = (
            (2 / 3)
            * wing_area**2
            * thickness_to_chord
            * (1.0 - taper_ratio / (1.0 + taper_ratio) ** 2)
            / span
        )
        fuel_cap_wing = fuel_density * volume_fraction * volume_of_wing

        outputs[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY] = fuel_cap_wing

    def compute_partials(self, inputs, partials):
        wing_area = inputs[Aircraft.Wing.AREA]

        fuel_density = inputs[Aircraft.Fuel.DENSITY]
        volume_fraction = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        span = inputs[Aircraft.Wing.SPAN]
        taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
        thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

        den = 1.0 + taper_ratio
        tr_fact = 1.0 - taper_ratio / den**2
        dfact = -1.0 / den**2 + 2.0 * taper_ratio / den**3

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Fuel.DENSITY] = (
            volume_fraction * (2 / 3) * wing_area**2 * thickness_to_chord * tr_fact / span
        )

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            fuel_density * (2 / 3) * wing_area**2 * thickness_to_chord * tr_fact / span
        )

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Wing.SPAN] = (
            -fuel_density
            * volume_fraction
            * (2 / 3)
            * wing_area**2
            * thickness_to_chord
            * tr_fact
            / span**2
        )

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Wing.TAPER_RATIO] = (
            fuel_density
            * volume_fraction
            * (2 / 3)
            * wing_area**2
            * thickness_to_chord
            * dfact
            / span
        )

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Wing.THICKNESS_TO_CHORD] = (
            fuel_density * volume_fraction * (2 / 3) * wing_area**2 * tr_fact / span
        )

        partials[Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, Aircraft.Wing.AREA] = (
            2.0
            * fuel_density
            * volume_fraction
            * (2 / 3)
            * wing_area
            * thickness_to_chord
            * tr_fact
            / span
        )
