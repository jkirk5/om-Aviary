"""
Define subsystem builder for Aviary core energy.

Classes
-------
EnergyBuilder : the interface for a energy subsystem builder.

CoreEnergyBuilder : the interface for Aviary's core energy subsystem builder
"""

# import warnings

import numpy as np
import openmdao.api as om
from openmdao.core.system import System

from aviary.mission.utils import separate_reserve_phases
from aviary.subsystems.energy.fuel_summation import FuelSummationGroup
from aviary.subsystems.energy.fuel_tank_model import FuelTankModel
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft, Dynamic


class EnergyBuilder(SubsystemBuilder):
    """
    Base energy builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the EnergyBuilder object with a given name.
    """

    _default_name = 'energy'


class CoreEnergyBuilder(EnergyBuilder):
    """Core energy subsystem builder."""

    def __init__(
        self,
        name=None,
        meta_data=None,
        code_origin: LegacyCode = None,
        fuel_tank_models=None,
        **kwargs,
    ):
        super().__init__(name=name, meta_data=meta_data)

        if fuel_tank_models is None:
            fuel_tank_models = []

        if not isinstance(fuel_tank_models, (list, np.ndarray)):
            fuel_tank_models = [fuel_tank_models]

        self.fuel_types = fuel_types = []

        # validate tank models
        for tank in fuel_tank_models:
            if not isinstance(tank, FuelTankModel):
                raise UserWarning(
                    'Fuel tank provided to energy builder is not a FuelTankModel object'
                )
            if tank.fuel_type in fuel_types:
                raise UserWarning(
                    'Aviary can currently only support one fuel tank model per type of fuel, '
                    f'multiple tanks containing {tank.fuel_type} were provided.'
                )

            fuel_types.append(tank.fuel_type)

        self.fuel_tank_models = fuel_tank_models

    def build_pre_mission(
        self, aviary_inputs: AviaryValues | None = None, subsystem_options: dict | None = None
    ) -> None | System:
        if len(self.fuel_tank_models) == 1:
            return self.fuel_tank_models[0].build_pre_mission(
                aviary_inputs=aviary_inputs, subsystem_options=subsystem_options
            )
        else:
            group = om.Group()
            for fuel_tank in self.fuel_tank_models:
                group.add_subsystem(
                    fuel_tank.name,
                    fuel_tank.build_pre_mission(
                        aviary_inputs=aviary_inputs, subsystem_options=subsystem_options
                    ),
                )
            return group

    def build_mission(
        self, num_nodes, aviary_inputs=None, user_options=None, subsystem_options=None
    ):
        num_fuel_types = len(self.fuel_types)

        fuel_group = om.Group()
        fuel_fill_comp = om.ExecComp(
            'fuel_fill = (fuel_capacity - cumulative_fuel) / fuel_capacity',
            fuel_fill={'val': np.zeros((num_nodes, num_fuel_types)), 'units': 'unitless'},
            fuel_capacity={'val': np.zeros(num_fuel_types), 'units': 'lbm'},
            cumulative_fuel={'val': np.zeros((num_nodes, num_fuel_types)), 'units': 'lbm'},
        )

        fuel_group.add_subsystem(
            'fuel_fill_rate',
            subsys=fuel_fill_comp,
            promotes_inputs=[
                ('fuel_capacity', Aircraft.Fuel.MAX_CAPACITY_MASS),
                (
                    'cumulative_fuel',
                    Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED,
                ),
            ],
            promotes_outputs=[('fuel_Fill', Dynamic.Vehicle.FUEL_TANK_FILL_FRACTION)],
        )

        return fuel_group

    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return [Aircraft.Fuel.MAX_CAPACITY_MASS, Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED]

    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return [Dynamic.Vehicle.FUEL_TANK_FILL_FRACTION]

    def build_post_mission(
        self,
        aviary_inputs: AviaryValues | None = None,
        mission_info=None,
        subsystem_options: dict | None = None,
        phase_mission_bus_lengths: dict | None = None,
    ) -> None | System:
        energy_group = om.Group()

        for fuel_tank in self.fuel_tank_models:
            energy_group.add_subsystem(
                fuel_tank.name,
                fuel_tank.build_pre_mission(
                    aviary_inputs=aviary_inputs, subsystem_options=subsystem_options
                ),
            )

        fuelgroup = FuelSummationGroup(mission_info=mission_info)

        energy_group.add_subsystem('fuel_summation', fuelgroup, promotes=['*'])

        return energy_group

    # def report(self, prob, reports_folder, **kwargs):
    #     """
    #     Generate the report for Aviary core energy.

    #     Parameters
    #     ----------
    #     prob : AviaryProblem
    #         The AviaryProblem that will be used to generate the report
    #     reports_folder : Path
    #         Location of the subsystems_report folder this report will be placed in
    #     """

    def get_post_mission_bus_variables(
        self, aviary_inputs: AviaryValues | None = None, mission_info: dict | None = None
    ) -> dict:
        post_mission_bus = {}
        main_phases, reserve_phases = separate_reserve_phases(mission_info)

        post_mission_bus[main_phases[-1]] = {
            Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                'post_mission_name': f'{self.name}.main_mission_fuel.mission_fuel',
                'src_indices': om.slicer[-1, :],
            }
        }

        if reserve_phases:
            post_mission_bus[reserve_phases[0]] = {
                Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                    'post_mission_name': f'{self.name}.reserve_mission_fuel.fuel_initial',
                    'src_indices': om.slicer[0, :],
                }
            }

            post_mission_bus[reserve_phases[-1]] = {
                Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                    'post_mission_name': f'{self.name}.reserve_mission_fuel.fuel_final',
                    'src_indices': om.slicer[-1, :],
                }
            }

        return post_mission_bus

    def get_states(
        self,
        aviary_inputs: AviaryValues | None = None,
        user_options: dict | None = None,
        subsystem_options: dict | None = None,
    ) -> dict:
        num_fuel_types = len(self.fuel_types)
        mass_ref = wrapped_convert_units(user_options['mass_ref'], 'kg')
        if mass_ref is None:
            mass_ref = 1.0e4
        state_dict = {
            Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED: {
                'fix_initial': True,
                'fix_final': False,
                'lower': 0.0,
                'ref': mass_ref,
                'units': 'kg',
                'rate_source': Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_TOTAL,
                'shape': (num_fuel_types,),
                'val': np.zeros(num_fuel_types),
                # 'targets': Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED,
            }
        }

        return state_dict

    def get_linked_variables(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        # link cumulative fuel burn between phases
        return [Dynamic.Vehicle.CUMULATIVE_FUEL_BURNED]
