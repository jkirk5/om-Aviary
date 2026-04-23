import openmdao.api as om

from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


class ReferenceGeometryFactor(om.ExplicitComponent):
    """Compute projected fuselage cabin floor area as a nondimensional factor."""

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='m')
        add_aviary_input(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, units='m')

        self.add_output('reference_geometry_factor', units='unitless')

    def setup_partials(self):
        self.declare_partials(
            'reference_geometry_factor',
            [Aircraft.Fuselage.MAX_WIDTH, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH],
        )

    def compute(self, inputs, outputs):
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        projected_length = inputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH]

        rgf = (
            max_width * projected_length
        )  # CAEP10 equation divides by 1 m^2, normalizing rgf as nondimensional

        outputs['reference_geometry_factor'] = rgf

    def compute_partials(self, inputs, J):
        max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        projected_length = inputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH]

        J['reference_geometry_factor', Aircraft.Fuselage.MAX_WIDTH] = projected_length
        J['reference_geometry_factor', Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH] = max_width
