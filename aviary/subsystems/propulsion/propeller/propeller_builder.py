from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.propulsion.propeller.propeller_performance import (
    PropellerPerformance,
)
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class PropellerBuilder(SubsystemBuilderBase):
    """
    Define the builder for a propeller model using the Hamilton Standard methodology that
    provides methods to define the propeller subsystem's states, design variables,
    fixed values, initial guesses, and mass names. It also provides methods to build
    OpenMDAO systems for the pre-mission and mission computations of the subsystem,
    to get the constraints for the subsystem, and to preprocess the inputs for
    the subsystem.
    """

    def __init__(self, name='HS_propeller'):
        """Initializes the PropellerBuilder object with a given name."""
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return

    def build_mission(self, num_nodes, aviary_inputs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""
        return PropellerPerformance(num_nodes=num_nodes, aviary_options=aviary_inputs)

    def get_design_vars(self):
        """
        Design vars are only tested to see if they exist in pre_mission
        Returns a dictionary of design variables for the gearbox subsystem, where the keys are the
        names of the design variables, and the values are dictionaries that contain the units for
        the design variable, the lower and upper bounds for the design variable, and any
        additional keyword arguments required by OpenMDAO for the design variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """

        # TODO bounds are rough placeholders
        DVs = {
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR: {
                'units': 'unitless',
                'lower': 100,
                'upper': 200,
                #'val': 100,  # initial value
            },
            Aircraft.Engine.PROPELLER_DIAMETER: {
                'units': 'ft',
                'lower': 0.0,
                'upper': None,
                #'val': 8,  # initial value
            },
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT: {
                'units': 'unitless',
                'lower': 0.0,
                'upper': 0.5,
                #'val': 0.5,
            },
        }
        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Parameters are only tested to see if they exist in mission.
        The value doesn't change throughout the mission.
        Returns a dictionary of fixed values for the propeller subsystem, where the keys
        are the names of the fixed values, and the values are dictionaries that contain
        the fixed value for the variable, the units for the variable, and any additional
        keyword arguments required by OpenMDAO for the variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """
        parameters = {
            Aircraft.Engine.PROPELLER_TIP_MACH_MAX: {
                'val': 1.0,
                'units': 'unitless',
            },
            Aircraft.Engine.PROPELLER_TIP_SPEED_MAX: {
                'val': 0.0,
                'units': 'unitless',
            },
            Aircraft.Engine.PROPELLER_TIP_SPEED_MAX: {
                'val': 0.0,
                'units': 'ft/s',
            },
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT: {
                'val': 0.0,
                'units': 'unitless',
            },
            Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR: {
                'val': 0.0,
                'units': 'unitless',
            },
            Aircraft.Engine.PROPELLER_DIAMETER: {
                'val': 0.0,
                'units': 'ft',
            },
            Aircraft.Nacelle.AVG_DIAMETER: {
                'val': 0.0,
                'units': 'ft',
            },
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Engine.Gearbox.MASS]

    def get_outputs(self):
        return [
            Dynamic.Mission.SHAFT_POWER + '_out',
            Dynamic.Mission.SHAFT_POWER_MAX + '_out',
            Dynamic.Mission.RPM + '_out',
            Dynamic.Mission.TORQUE + '_out',
            Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL,
        ]