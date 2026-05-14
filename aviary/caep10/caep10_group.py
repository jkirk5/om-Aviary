import openmdao.api as om

from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.mass_points import mass_points_calc
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.caep10.specific_air_range import SpecificAirRangeGroup
from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class CAEP10EmissionsGroup(om.Group):
    def initialize(self):
        self.options.declare('aviary_options', types=AviaryValues, default=None)
        self.options.declare('subsystems', types=list, default=[])
        self.options.declare('subsystem_options', types=dict, default={})

    def setup(self):
        aviary_inputs = self.options['aviary_options']
        subsystems = self.options['subsystems']
        subsystem_options = self.options['subsystem_options']

        # high, mid, low mass points
        self.add_subsystem(
            'mass_points_calc',
            mass_points_calc,
            promotes_inputs=[Aircraft.Design.GROSS_MASS],
            promotes_outputs=['mass_high', 'mass_mid', 'mass_low'],
        )

        # vectorize mass points
        mass_mux = om.MuxComp(vec_size=3)
        mass_mux.add_var(Dynamic.Vehicle.MASS, val=1.0, units='lbm')
        self.add_subsystem(
            'mass_mux',
            mass_mux,
            promotes_inputs=[
                ('mass_0', 'mass_high'),
                ('mass_1', 'mass_mid'),
                ('mass_2', 'mass_low'),
            ],
            promotes_outputs=['*'],
        )

        if aviary_inputs is not None and Settings.EQUATIONS_OF_MOTION in aviary_inputs:
            equations_of_motion = aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION)
        else:
            raise UserWarning(
                'Equations of motion were not provided for CAEP 10 emissions modeling.'
            )

        if equations_of_motion is EquationsOfMotion.ENERGY_STATE:
            ode_class = EnergyStateODE
        elif equations_of_motion is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM:
            ode_class = SimpleCruiseODE
        else:
            raise UserWarning(
                f'Invalid equations of motion {equations_of_motion} provided for CAEP 10 emissions '
                'modeling.'
            )

        # create a sub-problem containing the cruise ODE, to optimize Mach & Alt
        ode_opt = om.Problem(
            model=ode_class(
                num_nodes=3,
                aviary_options=aviary_inputs,
                subsystems=subsystems,
                subsystem_options=subsystem_options,
            )
        )

        # ode_opt.set_solver_print(0)

        # ode_opt.model.add_design_var(
        #     Dynamic.Mission.ALTITUDE,
        #     lower=0.0,
        #     ref=10000,
        #     units='ft',
        # )
        # # ode_opt.model.add_design_var(
        # #     Dynamic.Atmosphere.MACH, lower=0.0, upper=1.0, ref=1.0, units='unitless'
        # # )

        # ode_opt.model.add_objective(
        #     Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, ref=-1, index=0
        # )

        # # We don't have access to the original problem's optimizer here, so we must pick one
        # # Check for which optimizers are available
        # try:
        #     from pyoptsparse import OPT
        # except ImportError:
        #     use_pyoptsparse = False
        # else:
        #     use_pyoptsparse = True

        # if use_pyoptsparse:
        #     # Use SNOPT if available, otherwise try IPOPT
        #     try:
        #         OPT('IPOPT')
        #     except Exception:
        #         pass
        #     else:
        #         optimizer = 'IPOPT'
        #     try:
        #         OPT('SNOPT')
        #     except Exception:
        #         pass
        #     else:
        #         optimizer = 'SNOPT'
        # else:
        #     optimizer = 'SLSQP'

        # if not use_pyoptsparse:
        #     driver = ode_opt.driver = om.ScipyOptimizeDriver()
        # else:
        #     driver = ode_opt.driver = om.pyOptSparseDriver()

        # driver.options['optimizer'] = optimizer
        # # TODO per-driver settings
        # # Print Options #
        # driver.opt_settings['iSumm'] = 0
        # driver.opt_settings['iPrint'] = 0
        # # Optimizer Settings #
        # driver.opt_settings['Major iterations limit'] = 10
        # driver.opt_settings['Major optimality tolerance'] = 1e-4
        # driver.opt_settings['Major feasibility tolerance'] = 1e-6

        # self.add_subsystem(
        #     'cruise_perf',
        #     CustomSubmodelComp(
        #         problem=ode_opt,
        #         inputs=['*'],
        #         outputs=[
        #             Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        #             Dynamic.Mission.VELOCITY,
        #         ],
        #     ),
        #     promotes_inputs=['*'],
        #     promotes_outputs=[
        #         Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        #         Dynamic.Mission.VELOCITY,
        #     ],
        # )
        # setup_model_options(ode_opt, aviary_inputs)

        # self.add_subsystem(
        #     'cruise_perf',
        #     ode_class(
        #         num_nodes=3,
        #         aviary_options=aviary_inputs,
        #         subsystems=subsystems,
        #         subsystem_options=subsystem_options,
        #     ),
        #     promotes=['*'],
        # )

        self.add_subsystem('sar_group', SpecificAirRangeGroup(), promotes=['*'])
        for i in range(0, 3):
            self.connect(Dynamic.Mission.VELOCITY, f'tas_{i + 1}', src_indices=om.slicer[i])
            self.connect(
                Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                f'w_f_{i + 1}',
                src_indices=om.slicer[i],
            )

        self.add_subsystem('rgf', ReferenceGeometryFactor(), promotes=['*'])
        self.add_subsystem('co2_eval', CO2EmissionsMetric(), promotes=['*'])

        # self.add_subsystem(
        #     'CO2_emissions_resid',
        #     om.ExecComp(
        #         'CO2_emissions_resid = CO2_emissions_factor_max - CO2_emissions_factor',
        #         C02_emissions_resid={'val': 0.0, 'units': 'kg/km'},
        #         CO2_emissions_factor_max={'val': 1.0, 'units': 'kg/km'},
        #         CO2_emissions_factor={'val': 1.0, 'units': 'kg/km'},
        #     ),
        #     promotes=['*'],
        # )

        # self.add_constraint('CO2_emissions_resid', lower=0.0, ref=1.0)


class CruisePerformance(om.ExplicitComponent):
    def setup(self):
        

class CustomSubmodelComp(om.SubmodelComp):
    def setup(self):
        super().setup()

        sub = self._subprob

        sub.set_val(Dynamic.Atmosphere.MACH, [0.82, 0.82, 0.82])
        sub.set_val(Dynamic.Mission.ALTITUDE, [9144, 9144, 9144], units='m')
        sub.set_val(Dynamic.Mission.ALTITUDE_RATE, [0, 0, 0], 'm/s')
        # sub.set_val(Dynamic.Mission.VELOCITY_RATE, [0, 0, 0], 'm/s**2')
        sub.set_val(Dynamic.Atmosphere.MACH_RATE, [0, 0, 0], '1/s')

        sub.final_setup()
