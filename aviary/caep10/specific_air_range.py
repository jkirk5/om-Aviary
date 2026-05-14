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
