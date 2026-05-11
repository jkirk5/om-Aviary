import numpy as np
import openmdao.api as om

from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings
from aviary.utils.aviary_values import AviaryValues

inv_sar_avg = om.ExecComp(
    'inv_sar_avg = ((1/sar_1) + (1/sar_2) + (1/sar_3)) / 3',
    sar_1={'val': 1.0, 'units': 'm/kg'},
    sar_2={'val': 1.0, 'units': 'm/kg'},
    sar_3={'val': 1.0, 'units': 'm/kg'},
    inv_sar_avg={'val': 1.0, 'units': 'kg/m'},
)
# TODO vectorize me?
sar_calc = om.ExecComp(
    'sar_1 = tas_1/-w_f_1',
    sar_1={'val': 1.0, 'units': 'm/kg'},
    tas_1={'val': 1.0, 'units': 'm/s'},
    w_f_1={'val': 1.0, 'units': 'kg/s'},
)
sar_calc.add_expr(
    'sar_2 = tas_2/-w_f_2',
    sar_2={'val': 1.0, 'units': 'm/kg'},
    tas_2={'val': 1.0, 'units': 'm/s'},
    w_f_2={'val': 1.0, 'units': 'kg/s'},
)
sar_calc.add_expr(
    'sar_3 = tas_3/-w_f_3',
    sar_3={'val': 1.0, 'units': 'm/kg'},
    tas_3={'val': 1.0, 'units': 'm/s'},
    w_f_3={'val': 1.0, 'units': 'kg/s'},
)

class SpecificAirRangeGroup(om.Group):
    def setup(self):
        # calculate SAR for each point
        self.add_subsystem('sar_calc', sar_calc, promotes_inputs=['*'], promotes_outputs=['*'])

        # calculate average SAR
        self.add_subsystem(
            'inverse_sar_average',
            inv_sar_avg,
            promotes_inputs=['sar_1', 'sar_2', 'sar_3'],
            promotes_outputs=['inv_sar_avg'],
        )



class SpecificAirRangeGroup(om.Group):
    def initialize(self):
        add_aviary_option(self, Settings.EQUATIONS_OF_MOTION)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare(
            'subsystems',
            types=list,
            desc='list of subsystem builder instances to be added to the ODE',
        )

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

    prob.model.add_subsystem(
    'mass_calc',
    mass_calc,
    promotes_inputs=[Aircraft.Design.GROSS_MASS],
    promotes_outputs=['mass_high', 'mass_mid', 'mass_low'],
)
mass_mux = om.MuxComp(vec_size=3)
mass_mux.add_var(Dynamic.Vehicle.MASS, val=1.0, units='lbm')
prob.model.add_subsystem(
    'mass_mux',
    mass_mux,
    promotes_inputs=[('mass_0', 'mass_high'), ('mass_1', 'mass_mid'), ('mass_2', 'mass_low')],
    promotes_outputs=['*'],
)
prob.model.add_subsystem('cruise_perf', ode, promotes=['*'])
prob.model.add_subsystem('sar_group', SpecificAirRangeGroup(), promotes=['*'])
for i in range(0, 3):
    prob.model.connect(Dynamic.Mission.VELOCITY, f'tas_{i + 1}', src_indices=om.slicer[i])
    prob.model.connect(
        Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        f'w_f_{i + 1}',
        src_indices=om.slicer[i],
    )

        # add cruise points
        self.add_subsystem(
            'cruise_performance',
            ode_class(
                num_nodes=3,
                aviary_options=self.options['aviary_inputs'],
                subsystems=self.options['subsystems'],
            ),
        )

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
