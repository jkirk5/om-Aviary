import openmdao.api as om

from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.mass_points import mass_points_calc
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor
from aviary.caep10.specific_air_range import SpecificAirRangeGroup
from aviary.mission.energy_state.ode.energy_state_ODE import EnergyStateODE
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.variables import Aircraft, Dynamic, Settings


class CAEP10Builder(SubsystemBuilder):
    _default_name = 'CAEP10_Emissions'

    def build_post_mission(
        self,
        aviary_inputs: AviaryValues | None = None,
        mission_info: dict | None = None,
        subsystem_options: dict | None = None,
        phase_mission_bus_lengths=None,
    ):
        caep10 = om.Group()

        # high, mid, low mass points
        caep10.add_subsystem(
            'mass_points_calc',
            mass_points_calc,
            promotes_inputs=[Aircraft.Design.GROSS_MASS],
            promotes_outputs=['mass_high', 'mass_mid', 'mass_low'],
        )

        # vectorize mass points
        mass_mux = om.MuxComp(vec_size=3)
        mass_mux.add_var(Dynamic.Vehicle.MASS, val=1.0, units='lbm')
        caep10.add_subsystem(
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

        caep10.add_subsystem(
            'cruise_perf',
            ode_class(aviary_options=aviary_inputs, subsystem_options=subsystem_options),
            promotes=['*'],
        )
        caep10.add_subsystem('sar_group', SpecificAirRangeGroup(), promotes=['*'])
        for i in range(0, 3):
            caep10.connect(Dynamic.Mission.VELOCITY, f'tas_{i + 1}', src_indices=om.slicer[i])
            caep10.connect(
                Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                f'w_f_{i + 1}',
                src_indices=om.slicer[i],
            )
        caep10.add_subsystem('rgf', ReferenceGeometryFactor(), promotes=['*'])
        caep10.add_subsystem('co2_eval', CO2EmissionsMetric(), promotes=['*'])

        caep10.add_subsystem(
            'CO2_emissions_resid',
            om.ExecComp(
                'CO2_emissions_resid = CO2_emissions_factor_max - CO2_emissions_factor',
                C02_emissions_resid={'val': 0.0, 'units': 'kg/km'},
                CO2_emissions_factor_max={'val': 1.0, 'units': 'kg/km'},
                CO2_emissions_factor={'val': 1.0, 'units': 'kg/km'},
            ),
        )

        caep10.add_constraint('CO2_emissions_resid', lower=0.0, ref=1.0)

        return caep10
