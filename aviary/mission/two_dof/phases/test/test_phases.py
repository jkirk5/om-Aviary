"""Test for some features when using an external subsystem in the mission."""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.missions.two_dof_default import phase_info as two_dof_phase_info
from aviary.core.aviary_problem import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.enums import PhaseType


class DynBuilder(SubsystemBuilder):
    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            'x': {
                'rate_source': 'x_dot',
            }
        }

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return om.ExecComp('x_dot = x**2 + x')


if __name__ == '__main__':
    unittest.main()
