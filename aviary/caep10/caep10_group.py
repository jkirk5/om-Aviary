import dymos as dm
import openmdao.api as om

from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.mass_points import mass_points_calc
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.caep10.specific_air_range import SpecificAirRangeGroup
from aviary.mission.energy_state_problem_configurator import EnergyStateProblemConfigurator
from aviary.mission.two_dof_problem_configurator import TwoDOFProblemConfigurator
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import setup_model_options, setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings
from aviary.variable_info.variable_meta_data import CoreMetaData

mission_info = {
    'cruise': {
        'user_options': {
            'num_segments': 1,
            'order': 3,
            # Mach: single constant value over the phase, picked by optimizer
            'mach_optimize': True,
            'mach_polynomial_order': 0,
            'mach_bounds': ((0.5, 1.0), 'unitless'),
            # Altitude: single constant value over the phase, picked by optimizer
            'altitude_optimize': True,
            'altitude_polynomial_order': 0,
            'altitude_bounds': ((10000, 50000), 'ft'),
            'mass_ref': (150000, 'lbm'),
            'throttle_enforcement': 'boundary_constraint',
            'time_initial_bounds': ((0, 0), 'min'),
            'time_duration_bounds': ((0, 1000), 'min'),
        },
        'initial_guesses': {
            'altitude': ([30000.0, 30000.0], 'ft'),
            'mach': ([0.8, 0.8], 'unitless'),
        },
    }
}


class CAEP10EmissionsGroup(om.Group):
    def initialize(self):
        self.options.declare('aviary_options', types=AviaryValues, default=None)
        self.options.declare('subsystems', types=list, default=[])
        self.options.declare('subsystem_options', types=dict, default={})
        self.options.declare('meta_data', types=dict, default=CoreMetaData)

    def setup(self):
        aviary_inputs = self.options['aviary_options']
        subsystems = self.options['subsystems']
        subsystem_options = self.options['subsystem_options']
        meta_data = self.options['meta_data']

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
            configurator = EnergyStateProblemConfigurator()
        elif equations_of_motion is EquationsOfMotion.TWO_DEGREES_OF_FREEDOM:
            configurator = TwoDOFProblemConfigurator()
        else:
            raise UserWarning(
                f'Invalid equations of motion {equations_of_motion} provided for CAEP 10 emissions '
                'modeling.'
            )

        # Build the cruise phase (mirrors AviaryGroup._get_phase / add_phases).
        phase_name = 'cruise'
        phase_info = mission_info[phase_name]
        phase_builder_cls = configurator.get_phase_builder(self, phase_name, phase_info)
        phase_object = phase_builder_cls.from_phase_info(
            phase_name,
            phase_info,
            subsystems,
            meta_data=meta_data,
        )
        phase = phase_object.build_phase(aviary_options=aviary_inputs)

        # Fill in defaults from the builder's user_options.
        full_options = phase_object.user_options.to_phase_info()
        phase_info['user_options'] = full_options

        configurator.set_phase_options(self, phase_name, 0, phase, full_options, self.comm)

        # Assemble external parameters from subsystems for setup_trajectory_params.
        # all_subsystem_options = phase_info.get('subsystem_options', {})
        external_parameters = {phase_name: {}}
        traj_timeseries_outputs = []
        for subsystem in subsystems:
            sub_opts = subsystem_options.get(subsystem.name, {})
            parameter_dict = subsystem.get_parameters(
                aviary_inputs=aviary_inputs,
                user_options=full_options,
                subsystem_options=sub_opts,
            )
            for parameter in sorted(parameter_dict):
                external_parameters[phase_name][parameter] = parameter_dict[parameter]

            traj_timeseries_outputs.extend(
                subsystem.get_timeseries(
                    aviary_inputs=aviary_inputs,
                    user_options=full_options,
                    subsystem_options=sub_opts,
                )
            )

        for ts in traj_timeseries_outputs:
            phase.add_timeseries_output(ts)

        # Build the sub-problem model
        ode_model = om.Group()
        traj = ode_model.add_subsystem('traj', dm.Trajectory())
        traj.add_phase(phase_name, phase)

        setup_trajectory_params(
            ode_model,
            traj,
            aviary_inputs,
            [phase_name],
            meta_data=meta_data,
            external_parameters=external_parameters,
        )

        # create a sub-problem containing the cruise trajectory, to optimize Mach & Alt
        ode_opt = om.Problem(model=ode_model)
        setup_model_options(ode_opt, aviary_inputs, meta_data=meta_data)

        ode_opt.set_solver_print(0)

        ode_opt.model.add_objective(Mission.Objectives.RANGE, ref=-1)
        # We don't have access to the original problem's optimizer here, so we must pick one
        # Check for which optimizers are available
        try:
            from pyoptsparse import OPT
        except ImportError:
            use_pyoptsparse = False
        else:
            use_pyoptsparse = True

        if use_pyoptsparse:
            # Use SNOPT if available, otherwise try IPOPT
            try:
                OPT('IPOPT')
            except Exception:
                pass
            else:
                optimizer = 'IPOPT'
            try:
                OPT('SNOPT')
            except Exception:
                pass
            else:
                optimizer = 'SNOPT'
        else:
            optimizer = 'SLSQP'

        if not use_pyoptsparse:
            driver = ode_opt.driver = om.ScipyOptimizeDriver()
        else:
            driver = ode_opt.driver = om.pyOptSparseDriver()

        driver.options['optimizer'] = optimizer
        # TODO per-driver settings
        # Print Options #
        driver.opt_settings['iSumm'] = 0
        driver.opt_settings['iPrint'] = 0
        # Optimizer Settings #
        driver.opt_settings['Major iterations limit'] = 10
        driver.opt_settings['Major optimality tolerance'] = 1e-4
        driver.opt_settings['Major feasibility tolerance'] = 1e-6

        self.add_subsystem(
            'cruise_perf',
            om.SubmodelComp(
                problem=ode_opt,
                inputs=['*'],
                outputs=[
                    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                    Dynamic.Mission.VELOCITY,
                ],
            ),
            promotes_inputs=['*'],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                Dynamic.Mission.VELOCITY,
            ],
        )
        setup_model_options(ode_opt, aviary_inputs)

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
