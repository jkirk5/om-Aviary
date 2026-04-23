import openmdao.api as om
from aviary.caep10.reference_geometry_factor import ReferenceGeometryFactor

from aviary.caep10.co2_evaluation import CO2EmissionsMetric
from aviary.caep10.specific_air_range import SpecificAirRangeGroup
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class CAEP10Builder(SubsystemBuilder):
    _default_name = 'CAEP10_Emissions'

    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        caep10 = om.Group()
        caep10.add_subsystem('specific_air_range', SpecificAirRangeGroup(), promotes=['*'])
        caep10.add_subsystem('reference_geometry_factor', ReferenceGeometryFactor(), promotes=['*'])
        caep10.add_subsystem('CO2_emissions_factor', CO2EmissionsMetric(), promotes=['*'])

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
