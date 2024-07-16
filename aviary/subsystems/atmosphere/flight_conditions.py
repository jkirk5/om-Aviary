import numpy as np
import openmdao.api as om

from aviary import constants
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class FlightConditions(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "input_speed_type",
            default=SpeedType.TAS,
            types=SpeedType,
            desc="defines input airspeed as equivalent airspeed, true airspeed, or mach number",
        )

    def setup(self):
        nn = self.options["num_nodes"]
        in_type = self.options["input_speed_type"]
        arange = np.arange(self.options["num_nodes"])

        self.add_input(
            Dynamic.Mission.DENSITY,
            val=np.zeros(nn),
            units="slug/ft**3",
            desc="density of air",
        )
        self.add_input(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=np.zeros(nn),
            units="ft/s",
            desc="speed of sound",
        )

        self.add_output(
            Dynamic.Mission.DYNAMIC_PRESSURE,
            val=np.zeros(nn),
            units="lbf/ft**2",
            desc="dynamic pressure",
        )

        if in_type is SpeedType.TAS:
            self.add_input(
                "TAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )
            self.add_output(
                Dynamic.Mission.EQUIVALENT_AIRSPEED,
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed",
            )
            self.add_output(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [Dynamic.Mission.DENSITY, "TAS"],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.MACH,
                [Dynamic.Mission.SPEED_OF_SOUND, "TAS"],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.EQUIVALENT_AIRSPEED,
                ["TAS", Dynamic.Mission.DENSITY],
                rows=arange,
                cols=arange,
            )
        elif in_type is SpeedType.EAS:
            self.add_input(
                Dynamic.Mission.EQUIVALENT_AIRSPEED,
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed at",
            )
            self.add_output(
                "TAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )
            self.add_output(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [Dynamic.Mission.DENSITY, Dynamic.Mission.EQUIVALENT_AIRSPEED],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.MACH,
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.EQUIVALENT_AIRSPEED,
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                "TAS",
                [Dynamic.Mission.DENSITY, Dynamic.Mission.EQUIVALENT_AIRSPEED],
                rows=arange,
                cols=arange,
            )
        elif in_type is SpeedType.MACH:
            self.add_input(
                Dynamic.Mission.MACH,
                val=np.zeros(nn),
                units="unitless",
                desc="mach number",
            )
            self.add_output(
                Dynamic.Mission.EQUIVALENT_AIRSPEED,
                val=np.zeros(nn),
                units="ft/s",
                desc="equivalent air speed",
            )
            self.add_output(
                "TAS",
                val=np.zeros(nn),
                units="ft/s",
                desc="true air speed",
            )

            self.declare_partials(
                Dynamic.Mission.DYNAMIC_PRESSURE,
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.MACH,
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                "TAS",
                [Dynamic.Mission.SPEED_OF_SOUND, Dynamic.Mission.MACH],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.EQUIVALENT_AIRSPEED,
                [
                    Dynamic.Mission.SPEED_OF_SOUND,
                    Dynamic.Mission.MACH,
                    Dynamic.Mission.DENSITY,
                ],
                rows=arange,
                cols=arange,
            )

    def compute(self, inputs, outputs):

        in_type = self.options["input_speed_type"]

        rho = inputs[Dynamic.Mission.DENSITY]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        if in_type is SpeedType.TAS:
            TAS = inputs["TAS"]
            outputs[Dynamic.Mission.MACH] = mach = TAS / sos
            outputs[Dynamic.Mission.EQUIVALENT_AIRSPEED] = (
                TAS * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = 0.5 * rho * TAS**2

        elif in_type is SpeedType.EAS:
            EAS = inputs[Dynamic.Mission.EQUIVALENT_AIRSPEED]
            outputs["TAS"] = TAS = EAS / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            outputs[Dynamic.Mission.MACH] = mach = TAS / sos
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = (
                0.5 * EAS**2 * constants.RHO_SEA_LEVEL_ENGLISH
            )

        elif in_type is SpeedType.MACH:
            mach = inputs[Dynamic.Mission.MACH]
            outputs["TAS"] = TAS = sos * mach
            outputs[Dynamic.Mission.EQUIVALENT_AIRSPEED] = EAS = (
                TAS * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = 0.5 * rho * sos**2 * mach**2

    def compute_partials(self, inputs, J):
        in_type = self.options["input_speed_type"]

        rho = inputs[Dynamic.Mission.DENSITY]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        if in_type is SpeedType.TAS:
            TAS = inputs["TAS"]

            J[Dynamic.Mission.DYNAMIC_PRESSURE, "TAS"] = rho * TAS
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.DENSITY] = 0.5 * TAS**2

            J[Dynamic.Mission.MACH, "TAS"] = 1 / sos
            J[Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND] = -TAS / sos**2

            J[Dynamic.Mission.EQUIVALENT_AIRSPEED, "TAS"] = (
                rho / constants.RHO_SEA_LEVEL_ENGLISH
            ) ** 0.5
            J[Dynamic.Mission.EQUIVALENT_AIRSPEED, Dynamic.Mission.DENSITY] = (
                TAS * 0.5 * (rho ** (-0.5) / constants.RHO_SEA_LEVEL_ENGLISH**0.5)
            )

        elif in_type is SpeedType.EAS:
            EAS = inputs[Dynamic.Mission.EQUIVALENT_AIRSPEED]
            TAS = EAS / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5

            dTAS_dRho = -0.5 * EAS * constants.RHO_SEA_LEVEL_ENGLISH**0.5 / rho**1.5
            dTAS_dEAS = 1 / (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5

            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.EQUIVALENT_AIRSPEED] = (
                EAS * constants.RHO_SEA_LEVEL_ENGLISH
            )
            J[Dynamic.Mission.MACH, Dynamic.Mission.EQUIVALENT_AIRSPEED] = (
                dTAS_dEAS / sos
            )
            J[Dynamic.Mission.MACH, Dynamic.Mission.DENSITY] = dTAS_dRho / sos
            J[Dynamic.Mission.MACH, Dynamic.Mission.SPEED_OF_SOUND] = -TAS / sos**2
            J["TAS", Dynamic.Mission.DENSITY] = dTAS_dRho
            J["TAS", Dynamic.Mission.EQUIVALENT_AIRSPEED] = dTAS_dEAS

        elif in_type is SpeedType.MACH:
            mach = inputs[Dynamic.Mission.MACH]
            TAS = sos * mach

            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.SPEED_OF_SOUND] = (
                rho * sos * mach**2
            )
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH] = (
                rho * sos**2 * mach
            )
            J[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.DENSITY] = (
                0.5 * sos**2 * mach**2
            )
            J["TAS", Dynamic.Mission.SPEED_OF_SOUND] = mach
            J["TAS", Dynamic.Mission.MACH] = sos
            J[Dynamic.Mission.EQUIVALENT_AIRSPEED, Dynamic.Mission.SPEED_OF_SOUND] = (
                mach * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            J[Dynamic.Mission.EQUIVALENT_AIRSPEED, Dynamic.Mission.MACH] = (
                sos * (rho / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5
            )
            J[Dynamic.Mission.EQUIVALENT_AIRSPEED, Dynamic.Mission.DENSITY] = (
                TAS * (1 / constants.RHO_SEA_LEVEL_ENGLISH) ** 0.5 * 0.5 * rho ** (-0.5)
            )
