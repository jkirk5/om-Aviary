import unittest

from aviary.caep10.caep10_builder import CAEP10Builder
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilder
from aviary.utils.aviary_values import AviaryValues


class TestCAEP10Builder(TestSubsystemBuilder):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CAEP10Builder()
        self.aviary_values = AviaryValues()


if __name__ == '__main__':
    unittest.main()
