import numpy as np
import openmdao.api as om

from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

inv_sar_avg = om.ExecComp(
    'inv_sar_avg = ((1/sar_1) + (1/sar_2) + (1/sar_3)) / 3',
    sar_1={'val': 1.0, 'units': 'm/lbm'},
    sar_2={'val': 1.0, 'units': 'm/lbm'},
    sar_3={'val': 1.0, 'units': 'm/lbm'},
    inv_sar_avg={'val': 1.0, 'units': 'kg/m'},
)
# TODO vectorize me across three SARs - can't make copies of same component
sar_calc = om.ExecComp(
    'sar = tas/w_f',
    sar={'val': 1.0, 'units': 'm/lbm'},
    tas={'val': 1.0, 'units': 'm/s'},
    w_f={'val': 1.0, 'units': 'lbm/s'},
)


class SpecificAirRangeGroup(om.Group):
    def setup(self):
        # calculate masses for each evaluation point
        mass_calc = om.ExecComp()
        mass_calc.add_expr(
            'mass_high = 0.92 * togm',
            mass_high={'val': np.ones(1), 'units': 'lbm'},
            togm={'val': 1.0, 'units': 'lbm'},
        )
        mass_calc.add_expr(
            'mass_low = (0.45 * togm) + (0.63 * power(togm, 0.924))',
            mass_low={'val': np.ones(1), 'units': 'lbm'},
            togm={'val': 1.0, 'units': 'lbm'},
        )
        mass_calc.add_expr(
            'mass_mid = (mass_high + mass_low) / 2',
            mass_high={'val': np.ones(1), 'units': 'lbm'},
            mass_low={'val': 1.0, 'units': 'lbm'},
        )

        self.add_subsystem(
            'mass_calc',
            mass_calc,
            promotes_inputs=[('togm', Aircraft.Design.GROSS_MASS)],
            promotes_outputs=['mass_high', 'mass_mid', 'mass_low'],
        )

        # add cruise points for points 1 through 3
        self.add_subsystem('cruise_performance', CruisePoints())

        # calculate SAR for each point
        self.add_subsystem(
            'sar_1',
            sar_calc,
            promotes_outputs=[('sar', 'sar_1'), 'mass_high', 'mass_mid', 'mass_low'],
        )
        self.connect(
            f'cruise_performance.sar_1.{Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL}',
            'sar_1.w_f',
        )

        self.add_subsystem('sar_2', sar_calc, promotes_outputs=[('sar', 'sar_2')])
        self.connect(
            f'cruise_performance.sar_2.{Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL}',
            'sar_2.w_f',
        )

        self.add_subsystem('sar_3', sar_calc, promotes_outputs=[('sar', 'sar_3')])
        self.connect(
            f'cruise_performance.sar_3.{Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL}',
            'sar_3.w_f',
        )

        # calculate average SAR
        self.add_subsystem(
            'inverse_sar_average',
            inv_sar_avg,
            promotes_inputs=['sar_1', 'sar_2', 'sar_3'],
            promotes_outputs=['inv_sar_avg'],
        )


class CruisePoints(om.Group):
    """Analyze cruise performance at the 3 required points."""

    def initialize(self):
        add_aviary_option(self, Settings.EQUATIONS_OF_MOTION)

    def setup(self):
        equations_of_motion = self.options[Settings.EQUATIONS_OF_MOTION]

        if equations_of_motion is EquationsOfMotion.ENERGY_STATE:
            ode_class = EnergyStateODE
        elif equations_of_motion is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM:
            ode_class = SimpleCruiseODE
        else:
            raise UserWarning(
                f'Invalid equations of motion {equations_of_motion} provided for CAEP 10 emissions '
                'modeling.'
            )

        self.add_subsystem(
            'sar_1',
            ode_class(),
            promotes_inputs=[(Dynamic.Vehicle.MASS, 'mass_high')],  # , Dynamic.Atmosphere.MACH],
            # promotes_outputs=[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL],
        )
        self.add_subsystem(
            'sar_2',
            ode_class(),
            promotes_inputs=[(Dynamic.Vehicle.MASS, 'mass_mid')],  # , Dynamic.Atmosphere.MACH],
            # promotes_outputs=[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL],
        )
        self.add_subsystem(
            'sar_3',
            ode_class(),
            promotes_inputs=[(Dynamic.Vehicle.MASS, 'mass_low')],  # , Dynamic.Atmosphere.MACH],
            # promotes_outputs=[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL],
        )
