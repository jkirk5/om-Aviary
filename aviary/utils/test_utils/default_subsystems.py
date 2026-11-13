from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.energy.energy_builder import CoreEnergyBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.performance.performance_builder import CorePerformanceBuilder
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.subsystems.energy.flops_based.flops_tank import FuelTankFLOPS
from aviary.subsystems.energy.gasp_based.gasp_tank import FuelTankGASP


def get_default_subsystems(legacy_code, engines=None):
    """
    Get default premission subsystems.

    Arguments:
    ----------
    legacy_code : str, LegacyCode
        either FLOPS or GASP LegacyCode Enums, or their strings equivalents ('FLOPS', 'GASP')
    engine : <list of EngineDecks>
        List of EngineDecks
    """
    legacy_code = LegacyCode(legacy_code)
    prop = CorePropulsionBuilder('propulsion', CoreMetaData, engine_models=engines)
    mass = CoreMassBuilder('mass', CoreMetaData, legacy_code)
    aero = CoreAerodynamicsBuilder('aerodynamics', CoreMetaData, legacy_code)
    geom = CoreGeometryBuilder('geometry', CoreMetaData, legacy_code)
    perf = CorePerformanceBuilder('performance', CoreMetaData)
    if legacy_code is LegacyCode.FLOPS:
        fuel_tank = [FuelTankFLOPS()]
    elif legacy_code is LegacyCode.GASP:
        fuel_tank = [FuelTankGASP()]
    else:
        fuel_tank = []
    engy = CoreEnergyBuilder('energy', CoreMetaData, fuel_tank_models=fuel_tank)

    return {
        'propulsion': prop,
        'geometry': geom,
        'energy': engy,
        'mass': mass,
        'aerodynamics': aero,
        'performance': perf,
    }
