import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Aircraft

# calculate masses for each evaluation point
mass_calc_high = om.ExecComp()
mass_calc_high.add_expr(
    'mass_high = 0.92 * togm',
    mass_high={'val': np.ones(1), 'units': 'kg'},
    togm={'val': 1.0, 'units': 'kg'},
)
mass_calc_low = om.ExecComp()
mass_calc_low.add_expr(
    'mass_low = (0.45 * togm) + (0.63 * power(togm, 0.924))',
    mass_low={'val': np.ones(1), 'units': 'kg'},
    togm={'val': 1.0, 'units': 'kg'},
)
mass_calc_mid = om.ExecComp()
mass_calc_mid.add_expr(
    'mass_mid = (mass_high + mass_low) / 2',
    mass_mid={'val': np.ones(1), 'units': 'kg'},
    mass_high={'val': np.ones(1), 'units': 'kg'},
    mass_low={'val': np.ones(1), 'units': 'kg'},
)
mass_points_calc = om.Group()
mass_points_calc.add_subsystem(
    'mass_calc_high',
    mass_calc_high,
    promotes_inputs=[('togm', Aircraft.Design.GROSS_MASS)],
    promotes_outputs=['*'],
)
mass_points_calc.add_subsystem(
    'mass_calc_low',
    mass_calc_low,
    promotes_inputs=[('togm', Aircraft.Design.GROSS_MASS)],
    promotes_outputs=['*'],
)
mass_points_calc.add_subsystem(
    'mass_calc_mid', mass_calc_mid, promotes_inputs=['*'], promotes_outputs=['*']
)
