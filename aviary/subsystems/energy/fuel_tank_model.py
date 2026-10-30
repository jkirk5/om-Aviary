"""
Define utilities for building fuel tank models.

Classes
-------
TankModel : the interface for a fuel tank model builder.
"""

from openmdao.core.system import System

from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import FuelType


class FuelTankModel(SubsystemBuilder):
    """
    Define the interface for a fuel tank model builder.

    Attributes
    ----------
    name : str ('fuel_tank')
        object label.
    options : AviaryValues (<empty>)
        inputs and options related to engine model.

    Methods
    -------
    build_pre_mission
    build_mission
    build_post_mission
    get_val
    get_item
    set_val
    update
    """

    fuel_type = FuelType.JET_A

    def __init__(
        self,
        name: str | None = None,
        meta_data: dict | None = None,
        **kwargs,
    ):
        # default name is dynamically set based on its fuel type
        # need to check inherrited class did not define their own default name
        if name is None and self._default_name == SubsystemBuilder._default_name:
            name = f'{self.fuel_type.name.lower()}_fuel_tank'

        super().__init__(name, meta_data=meta_data)

    # def build_pre_mission(
    #     self, aviary_inputs: AviaryValues | None = None, subsystem_options: dict | None = None
    # ) -> None | System:
    #     if aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.FLOPS:
    #         pre_mission = om.Group()
    #         return pre_mission
    #     elif aviary_inputs.get_val(Settings.MASS_METHOD) is LegacyCode.GASP:
    #         pre_mission = om.Group()
    #         return pre_mission
    #     else:
    #         return None

    def build_mission(
        self,
        num_nodes: int,
        aviary_inputs: AviaryValues | None = None,
        user_options: dict | None = None,
        subsystem_options: dict | None = None,
    ) -> None | System:
        return super().build_mission(num_nodes, aviary_inputs, user_options, subsystem_options)
