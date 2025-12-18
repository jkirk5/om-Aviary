from copy import deepcopy

from aviary.models.aircraft.test_aircraft.GASP_multiengine import engines, inputs
from aviary.models.missions.two_dof_default import phase_info

import openmdao.api as om

from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.utils.functions import set_aviary_initial_values
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft, Mission
import aviary.api as av

# local_phase_info = deepcopy(phase_info)

# prob = av.AviaryProblem()

# # Load aircraft and options data from provided sources
# prob.load_inputs(inputs, phase_info, engines)

# prob.check_and_preprocess_inputs()

# prob.build_model()

# # optimizer and iteration limit are optional provided here
# prob.add_driver('SLSQP', max_iter=0)

# prob.add_design_variables()

# prob.add_objective()

# prob.setup()

# prob.run_aviary_problem()
GASP = LegacyCode.GASP
prob = om.Problem()

prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engines)
mass = CoreMassBuilder('core_mass', BaseMetaData, GASP)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, GASP)
geom = CoreGeometryBuilder(
    'core_geometry',
    BaseMetaData,
    code_origin=(GASP),
    code_origin_to_prioritize=GASP,
)

core_subsystems = [prop]  # , geom, mass, aero]

prob.model.add_subsystem(
    'pre_mission',
    CorePreMission(aviary_options=inputs, subsystems=core_subsystems),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

prob.model.engine_builders = engines

inputs.delete(Aircraft.Engine.SCALED_SLS_THRUST)

setup_model_options(prob, inputs)

prob.setup(check=False, force_alloc_complex=True)

# Initial guess for gross mass.
# We set it to an unconverged value to test convergence.
# prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

# Set initial values for all variables.
set_aviary_initial_values(prob, inputs)

prob.run_model()
