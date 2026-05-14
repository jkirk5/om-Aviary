from openmdao.core.system import System

from aviary.subsystems.performance.performance_premission import DesignMetrics
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class PerformanceBuilder(SubsystemBuilder):
    """
    Base performance builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None, subsystems=None):
        Initializes the PerformanceBuilder object with a given name, and stores list of subsystems.
    """

    _default_name = 'performance'

    def __init__(self, name=None, meta_data=None, subsystems=None):
        super().__init__(name=name, meta_data=meta_data)
        self.subsystems = subsystems


class CorePerformanceBuilder(PerformanceBuilder):
    """Core performance analysis subsystem builder."""

    def build_pre_mission(self, aviary_inputs, subsystem_options=None) -> None | System:
        # currently only calculating T/W & W/S. Other Performance systems TBD
        return DesignMetrics(subsystems=self.subsystems)

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ) -> None | System:
        return
