import numpy as np

import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.functions import sigmoidX, dSigmoidXdx
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft, Mission, Settings


class BodyTankCalculations(om.ExplicitComponent):
    """
    Computation of fuel capacity of the auxiliary tank, extra required design fuel volume
    along with mass of fuel in it, and minimum wing fuel mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):

        add_aviary_input(self, Aircraft.Fuel.WING_VOLUME_DESIGN, units='ft**3')
        add_aviary_input(self, Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, units='ft**3')
        self.add_input("fuel_mass_min", val=2000, units="lbm",
                       desc="WFAMIN: minimum value of fuel mass (set when max payload is carried)")
        add_aviary_input(self, Mission.Design.FUEL_MASS_REQUIRED, units='lbm')
        self.add_input("max_wingfuel_mass", val=6, units="lbm",
                       desc="WFWMX: maximum wingfuel mass")
        add_aviary_input(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3')
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units="lbm/ft**3")
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Mission.Design.FUEL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.OPERATING_MASS, units='lbm')

        # WFXTRA: extra amount of fuel that is required but does not fit in wings
        add_aviary_output(self, Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, units='lbm')
        self.add_output("extra_fuel_volume", val=0, units="ft**3",
                        desc="FVOLXTRA: excess required design fuel volume (including fuel margin) greater than geometric fuel volume of wings")
        self.add_output("max_extra_fuel_mass", val=0, units="lbm",
                        desc="WFXTRAMX: mass of fuel that fits in extra_fuel_volume")
        self.add_output("wingfuel_mass_min", val=0, units="lbm",
                        desc="WFWMIN: minimum wing fuel mass")
        add_aviary_output(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')

        self.declare_partials(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, [
                              Mission.Design.FUEL_MASS_REQUIRED, "max_wingfuel_mass"])
        self.declare_partials(
            "extra_fuel_volume",
            [Aircraft.Fuel.WING_VOLUME_DESIGN, Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX],
        )
        self.declare_partials(
            "max_extra_fuel_mass",
            [
                Aircraft.Fuel.WING_VOLUME_DESIGN,
                Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                Aircraft.Fuel.DENSITY,
            ],
        )
        self.declare_partials(
            "wingfuel_mass_min",
            [
                "fuel_mass_min",
                Aircraft.Fuel.WING_VOLUME_DESIGN,
                Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                Aircraft.Fuel.DENSITY,
            ],
        )
        self.declare_partials(
            Aircraft.Fuel.TOTAL_CAPACITY,
            [
                Mission.Design.FUEL_MASS,
                Mission.Design.FUEL_MASS_REQUIRED,
                "max_wingfuel_mass",
                Mission.Design.GROSS_MASS,
                Aircraft.Design.OPERATING_MASS,
            ],
        )

    def compute(self, inputs, outputs):

        design_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_DESIGN]
        max_wingfuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX]
        fuel_wt_min = inputs["fuel_mass_min"] * GRAV_ENGLISH_LBM
        req_fuel_wt = inputs[Mission.Design.FUEL_MASS_REQUIRED] * \
            GRAV_ENGLISH_LBM
        max_wingfuel_wt = inputs["max_wingfuel_mass"] * GRAV_ENGLISH_LBM
        geometric_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fuel_wt_des = inputs[Mission.Design.FUEL_MASS] * GRAV_ENGLISH_LBM
        OEW = inputs[Aircraft.Design.OPERATING_MASS] * GRAV_ENGLISH_LBM

        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        extra_fuel_volume = sigmoidX(design_fuel_vol - max_wingfuel_vol, 0) * (
            design_fuel_vol - geometric_fuel_vol
        )

        verbosity = self.options[Settings.VERBOSITY]
        if verbosity >= Verbosity.BRIEF:
            if (req_fuel_wt > max_wingfuel_wt) and (design_fuel_vol > max_wingfuel_vol):
                print("Warning: req_fuel_mass > max_wingfuel_mass, adding a body tank")
            if (req_fuel_wt < max_wingfuel_wt) and (design_fuel_vol > max_wingfuel_vol):
                print("Warning: design_fuel_vol > max_wingfuel_vol, adding a body tank")

        extra_fuel_wt = req_fuel_wt - max_wingfuel_wt
        if smooth:
            extra_fuel_wt = extra_fuel_wt * sigmoidX(extra_fuel_wt, 0, 1 / 50)
        else:
            if extra_fuel_wt < 0:
                extra_fuel_wt = 0
        max_extra_fuel_wt = extra_fuel_volume * rho_fuel
        wingfuel_wt_min = fuel_wt_min - max_extra_fuel_wt
        wingfuel_wt_min = wingfuel_wt_min * sigmoidX(wingfuel_wt_min, 0)
        max_fuel_avail_est = fuel_wt_des + extra_fuel_wt
        max_fuel_avail_new = gross_wt_initial - OEW
        est_GTOW = OEW + max_fuel_avail_est
        max_fuel_avail = max_fuel_avail_est * sigmoidX(
            gross_wt_initial - est_GTOW, 0, 1. / 110.0
        ) + max_fuel_avail_new * sigmoidX(est_GTOW - gross_wt_initial, 0, 1 / 110.0)

        outputs[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY] = \
            extra_fuel_wt / GRAV_ENGLISH_LBM
        outputs["extra_fuel_volume"] = extra_fuel_volume
        outputs["max_extra_fuel_mass"] = max_extra_fuel_wt / GRAV_ENGLISH_LBM
        outputs["wingfuel_mass_min"] = wingfuel_wt_min / GRAV_ENGLISH_LBM
        outputs[Aircraft.Fuel.TOTAL_CAPACITY] = \
            max_fuel_avail / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):

        design_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_DESIGN]
        max_wingfuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX]
        fuel_wt_min = inputs["fuel_mass_min"] * GRAV_ENGLISH_LBM
        req_fuel_wt = inputs[Mission.Design.FUEL_MASS_REQUIRED] * \
            GRAV_ENGLISH_LBM
        max_wingfuel_wt = inputs["max_wingfuel_mass"] * GRAV_ENGLISH_LBM
        geometric_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fuel_wt_des = inputs[Mission.Design.FUEL_MASS] * GRAV_ENGLISH_LBM
        OEW = inputs[Aircraft.Design.OPERATING_MASS] * GRAV_ENGLISH_LBM

        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        extra_fuel_volume = sigmoidX(design_fuel_vol - max_wingfuel_vol, 0) * (
            design_fuel_vol - geometric_fuel_vol
        )
        max_extra_fuel_wt = extra_fuel_volume * rho_fuel

        extra_fuel_wt = req_fuel_wt - max_wingfuel_wt
        if smooth:
            extra_fuel_wt = extra_fuel_wt * sigmoidX(extra_fuel_wt, 0, 1 / 50.0)
        else:
            if extra_fuel_wt < 0:
                extra_fuel_wt = 0
        max_fuel_avail_est = fuel_wt_des + extra_fuel_wt
        max_fuel_avail_new = gross_wt_initial - OEW
        est_GTOW = OEW + max_fuel_avail_est

        if smooth:
            dExtraFuelWt_dReqFuelWt = (
                sigmoidX(req_fuel_wt - max_wingfuel_wt, 0, 1 / 50.0)
                + (req_fuel_wt - max_wingfuel_wt)
                * dSigmoidXdx(req_fuel_wt - max_wingfuel_wt, 0, 1 / 50.0)
                * 1
                / 50
            )
            dExtraFuelWt_dMaxWingfuelWt = -1 * sigmoidX(
                (req_fuel_wt - max_wingfuel_wt), 0, 1 / 50.0
            ) + (req_fuel_wt - max_wingfuel_wt) * dSigmoidXdx(
                req_fuel_wt - max_wingfuel_wt, 0, 1 / 50.0
            ) * (
                -1 / 50
            )
        else:
            dExtraFuelWt_dReqFuelWt = 1 if extra_fuel_wt > 0 else 0
            dExtraFuelWt_dMaxWingfuelWt = -1 if extra_fuel_wt > 0 else 0
        J[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            Mission.Design.FUEL_MASS_REQUIRED] = dExtraFuelWt_dReqFuelWt
        J[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            "max_wingfuel_mass"] = dExtraFuelWt_dMaxWingfuelWt

        J["extra_fuel_volume", Aircraft.Fuel.WING_VOLUME_DESIGN] = dSigmoidXdx(
            design_fuel_vol - max_wingfuel_vol, 0
        ) * (design_fuel_vol - geometric_fuel_vol) + sigmoidX(
            design_fuel_vol - max_wingfuel_vol, 0
        )
        J["extra_fuel_volume", Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX] = (
            dSigmoidXdx(design_fuel_vol - max_wingfuel_vol, 0)
            * (-1)
            * (design_fuel_vol - geometric_fuel_vol)
        )
        J["extra_fuel_volume", Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = -sigmoidX(
            design_fuel_vol - max_wingfuel_vol, 0
        )

        dMEFWt_dDesFuelVol = (
            dSigmoidXdx(design_fuel_vol - max_wingfuel_vol, 0)
            * (design_fuel_vol - geometric_fuel_vol)
            + sigmoidX(design_fuel_vol - max_wingfuel_vol, 0)
        ) * rho_fuel
        dMEFWt_dMaxWingfuelVol = (
            dSigmoidXdx(design_fuel_vol - max_wingfuel_vol, 0)
            * (-1)
            * (design_fuel_vol - geometric_fuel_vol)
            * rho_fuel
        )
        dMEFWt_dGeomFuelVol = (
            -sigmoidX(design_fuel_vol - max_wingfuel_vol, 0) * rho_fuel
        )

        J["max_extra_fuel_mass", Aircraft.Fuel.WING_VOLUME_DESIGN] = \
            dMEFWt_dDesFuelVol / GRAV_ENGLISH_LBM
        J["max_extra_fuel_mass", Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX] = \
            dMEFWt_dMaxWingfuelVol / GRAV_ENGLISH_LBM
        J["max_extra_fuel_mass", Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = \
            dMEFWt_dGeomFuelVol / GRAV_ENGLISH_LBM,
        J["max_extra_fuel_mass", Aircraft.Fuel.DENSITY] = dMEFWt_dRhoFuel = extra_fuel_volume

        J["wingfuel_mass_min", "fuel_mass_min"] = sigmoidX(fuel_wt_min - max_extra_fuel_wt, 0) + (
            fuel_wt_min - max_extra_fuel_wt
        ) * dSigmoidXdx(fuel_wt_min - max_extra_fuel_wt, 0)
        J["wingfuel_mass_min", Aircraft.Fuel.WING_VOLUME_DESIGN] = ((
            fuel_wt_min - max_extra_fuel_wt
        ) * dSigmoidXdx(fuel_wt_min - max_extra_fuel_wt, 0) * (
            -dMEFWt_dDesFuelVol
        ) - dMEFWt_dDesFuelVol * sigmoidX(
            fuel_wt_min - max_extra_fuel_wt, 0
        )) / GRAV_ENGLISH_LBM
        J["wingfuel_mass_min", Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX] = (fuel_wt_min * dSigmoidXdx(
            fuel_wt_min - max_extra_fuel_wt, 0
        ) * (-dMEFWt_dMaxWingfuelVol) - (
            dMEFWt_dMaxWingfuelVol * sigmoidX(fuel_wt_min - max_extra_fuel_wt, 0)
            + max_extra_fuel_wt
            * dSigmoidXdx(fuel_wt_min - max_extra_fuel_wt, 0)
            * (-dMEFWt_dMaxWingfuelVol)
        )) / GRAV_ENGLISH_LBM
        J["wingfuel_mass_min", Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = (fuel_wt_min * dSigmoidXdx(
            fuel_wt_min - max_extra_fuel_wt, 0
        ) * (-dMEFWt_dGeomFuelVol) - (
            dMEFWt_dGeomFuelVol * sigmoidX(fuel_wt_min - max_extra_fuel_wt, 0)
            + max_extra_fuel_wt
            * dSigmoidXdx(fuel_wt_min - max_extra_fuel_wt, 0)
            * (-dMEFWt_dGeomFuelVol)
        )) / GRAV_ENGLISH_LBM
        J["wingfuel_mass_min", Aircraft.Fuel.DENSITY] = fuel_wt_min * dSigmoidXdx(
            fuel_wt_min - max_extra_fuel_wt, 0
        ) * (-dMEFWt_dRhoFuel) - (
            dMEFWt_dRhoFuel * sigmoidX(fuel_wt_min - max_extra_fuel_wt, 0)
            + max_extra_fuel_wt
            * dSigmoidXdx(fuel_wt_min - max_extra_fuel_wt, 0)
            * (-dMEFWt_dRhoFuel)
        )

        int1 = sigmoidX(gross_wt_initial - est_GTOW, 0, 1 / 110.0)
        int2 = sigmoidX(est_GTOW - gross_wt_initial, 0, 1 / 110.0)
        max_fuel_avail = max_fuel_avail_est * int1 + max_fuel_avail_new * int2

        dInt1_dFuelWtDes = dSigmoidXdx(
            gross_wt_initial - est_GTOW, 0, 1 / 110.0) * (-1 / 110)
        dInt1_dReqFuelWt = dSigmoidXdx(gross_wt_initial - est_GTOW, 0, 1 / 110.0) * (
            -dExtraFuelWt_dReqFuelWt
        )
        dInt1_dMaxWingfuelWt = dSigmoidXdx(gross_wt_initial - est_GTOW, 0, 1 / 110.0) * (
            -dExtraFuelWt_dMaxWingfuelWt
        )

        dInt2_dOEM = dSigmoidXdx(est_GTOW - gross_wt_initial, 0, 1 / 110.0) / 110
        dInt2_dFuelWtDes = dSigmoidXdx(est_GTOW - gross_wt_initial, 0, 1 / 110.0) / 110
        dInt2_dReqFuelWt = (
            dSigmoidXdx(est_GTOW - gross_wt_initial, 0, 1 / 110.0) *
            dExtraFuelWt_dReqFuelWt
        )
        dInt2_dMaxWingfuelWt = (
            dSigmoidXdx(est_GTOW - gross_wt_initial, 0, 1 / 110.0) *
            dExtraFuelWt_dMaxWingfuelWt
        )
        dInt2_dGTOW = dSigmoidXdx(est_GTOW - gross_wt_initial, 0, 1 / 110.0) * (-1 / 110)

        dMFAest_dFuelWtDes = 1
        dMFAest_dReqFuelWt = dExtraFuelWt_dReqFuelWt
        dMFAest_dMaxWingfuelWt = dExtraFuelWt_dMaxWingfuelWt

        J[Aircraft.Fuel.TOTAL_CAPACITY, Mission.Design.FUEL_MASS] = (
            int1
            + max_fuel_avail_est * dInt1_dFuelWtDes
            + max_fuel_avail_new * dInt2_dFuelWtDes
        )
        J[Aircraft.Fuel.TOTAL_CAPACITY, Mission.Design.FUEL_MASS_REQUIRED] = (
            dMFAest_dReqFuelWt * int1
            + max_fuel_avail_est * dInt1_dReqFuelWt
            + max_fuel_avail_new * dInt2_dReqFuelWt
        )
        J[Aircraft.Fuel.TOTAL_CAPACITY, "max_wingfuel_mass"] = (
            dMFAest_dMaxWingfuelWt * int1
            + max_fuel_avail_est * dInt1_dMaxWingfuelWt
            + max_fuel_avail_new * dInt2_dMaxWingfuelWt
        )
        J[Aircraft.Fuel.TOTAL_CAPACITY, Mission.Design.GROSS_MASS] = (
            max_fuel_avail_est * dInt2_dGTOW
            + 1 * int2
            + max_fuel_avail_new * dInt2_dGTOW
        )
        J[Aircraft.Fuel.TOTAL_CAPACITY, Aircraft.Design.OPERATING_MASS] = (
            max_fuel_avail_est * dInt2_dOEM - 1 * int2 + max_fuel_avail_new * dInt2_dOEM
        )


class FuelAndOEMOutputs(om.ExplicitComponent):
    """
    Computation of various fuel and OEM parameters (such as wing fuel mass when
    operating empty, wing tank fuel volume when carrying maximum fuel, wing tank
    fuel volume when carrying design fuel plus fuel margin, operating mass empty
    of the aircraft, allowable payload mass with maximum fuel, mass of wing fuel
    based on volume, maximum wingfuel mass, and wing tank volume based on maximum
    wing fuel weight).
    """

    def setup(self):

        add_aviary_input(self, Aircraft.Fuel.DENSITY, units="lbm/ft**3")
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Controls.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.FIXED_EQUIPMENT_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.FIXED_USEFUL_LOAD, units='lbm')
        add_aviary_input(self, Mission.Design.FUEL_MASS_REQUIRED, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, units='ft**3')
        add_aviary_input(self, Aircraft.Fuel.FUEL_MARGIN, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')

        self.add_output(
            "OEM_wingfuel_mass",
            val=0,
            units="lbm",
            desc="WFWOWE: wing fuel mass when operating empty",
        )
        self.add_output(
            "OEM_fuel_vol",
            val=0,
            units="ft**3",
            desc="FVOLW: wing tank fuel volume when carrying maximum fuel",
        )
        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_DESIGN, units='ft**3')
        add_aviary_output(self, Aircraft.Design.OPERATING_MASS, units='lbm')

        self.add_output(
            "payload_mass_max_fuel",
            val=0,
            units="lbm",
            desc="WPLMXF: allowable payload mass with maximum fuel",
        )
        self.add_output(
            "volume_wingfuel_mass",
            val=0,
            units="lbm",
            desc=" mass of wing fuel based on volume, sometimes set as WFWMX in GASP, depending on if it exceeds the OEM fuel mass",
        )
        self.add_output(
            "max_wingfuel_mass", val=0, units="lbm", desc="WFWMX: maximum wingfuel mass"
        )
        add_aviary_output(self, Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, units='ft**3')

        self.declare_partials(
            "OEM_wingfuel_mass",
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
            ],
        )
        self.declare_partials(
            "OEM_fuel_vol",
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                Aircraft.Fuel.DENSITY,
            ],
        )
        self.declare_partials(
            Aircraft.Fuel.WING_VOLUME_DESIGN, [Mission.Design.FUEL_MASS_REQUIRED,
                                               Aircraft.Fuel.DENSITY, Aircraft.Fuel.FUEL_MARGIN]
        )
        self.declare_partials(
            Aircraft.Design.OPERATING_MASS,
            [Aircraft.Propulsion.MASS, Aircraft.Controls.TOTAL_MASS, Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS, Aircraft.Design.FIXED_USEFUL_LOAD],
            val=1,
        )
        self.declare_partials(
            "payload_mass_max_fuel", [Mission.Design.GROSS_MASS], val=1
        )
        self.declare_partials(
            "payload_mass_max_fuel",
            [
                Aircraft.Fuel.TOTAL_CAPACITY,
                Aircraft.Propulsion.MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
            ],
            val=-1,
        )
        self.declare_partials(
            "volume_wingfuel_mass", [
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, Aircraft.Fuel.DENSITY]
        )
        self.declare_partials(
            "max_wingfuel_mass",
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                Aircraft.Fuel.DENSITY,
            ],
        )
        self.declare_partials(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
                Aircraft.Fuel.DENSITY,
            ],
        )

    def compute(self, inputs, outputs):

        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        propulsion_wt = inputs[Aircraft.Propulsion.MASS] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        struct_wt = inputs[Aircraft.Design.STRUCTURE_MASS] * GRAV_ENGLISH_LBM
        fixed_equip_wt = inputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] * \
            GRAV_ENGLISH_LBM
        useful_wt = inputs[Aircraft.Design.FIXED_USEFUL_LOAD] * \
            GRAV_ENGLISH_LBM
        req_fuel_wt = inputs[Mission.Design.FUEL_MASS_REQUIRED] * \
            GRAV_ENGLISH_LBM
        geometric_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX]
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]
        max_fuel_avail = inputs[Aircraft.Fuel.TOTAL_CAPACITY] * \
            GRAV_ENGLISH_LBM

        OEM_wingfuel_wt = (
            gross_wt_initial
            - propulsion_wt
            - control_wt
            - struct_wt
            - fixed_equip_wt
            - useful_wt
        )

        OEM_fuel_vol = OEM_wingfuel_wt / rho_fuel
        design_fuel_vol = (1.0 + fuel_margin / 100.0) * req_fuel_wt / rho_fuel

        OEW = propulsion_wt + control_wt + struct_wt + fixed_equip_wt + useful_wt

        volume_wingfuel_wt = geometric_fuel_vol * rho_fuel
        max_wingfuel_wt = OEM_wingfuel_wt * sigmoidX(
            volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0
        ) + volume_wingfuel_wt * sigmoidX(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
        payload_wt_max_fuel = gross_wt_initial - OEW - max_fuel_avail
        max_wingfuel_vol = max_wingfuel_wt / (rho_fuel)

        outputs["OEM_wingfuel_mass"] = OEM_wingfuel_wt / GRAV_ENGLISH_LBM
        outputs["OEM_fuel_vol"] = OEM_fuel_vol
        outputs[Aircraft.Fuel.WING_VOLUME_DESIGN] = design_fuel_vol
        outputs[Aircraft.Design.OPERATING_MASS] = OEW / GRAV_ENGLISH_LBM
        outputs["payload_mass_max_fuel"] = \
            payload_wt_max_fuel / GRAV_ENGLISH_LBM
        outputs["volume_wingfuel_mass"] = volume_wingfuel_wt / GRAV_ENGLISH_LBM
        outputs["max_wingfuel_mass"] = max_wingfuel_wt / GRAV_ENGLISH_LBM
        outputs[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX] = max_wingfuel_vol

    def compute_partials(self, inputs, J):

        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        propulsion_wt = inputs[Aircraft.Propulsion.MASS] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        struct_wt = inputs[Aircraft.Design.STRUCTURE_MASS] * GRAV_ENGLISH_LBM
        fixed_equip_wt = inputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] * \
            GRAV_ENGLISH_LBM
        useful_wt = inputs[Aircraft.Design.FIXED_USEFUL_LOAD] * \
            GRAV_ENGLISH_LBM
        req_fuel_wt = inputs[Mission.Design.FUEL_MASS_REQUIRED] * \
            GRAV_ENGLISH_LBM
        geometric_fuel_vol = inputs[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX]
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]
        max_fuel_avail = inputs[Aircraft.Fuel.TOTAL_CAPACITY] * \
            GRAV_ENGLISH_LBM

        OEM_wingfuel_wt = (
            gross_wt_initial
            - propulsion_wt
            - control_wt
            - struct_wt
            - fixed_equip_wt
            - useful_wt
        )
        volume_wingfuel_wt = geometric_fuel_vol * rho_fuel
        max_wingfuel_wt = OEM_wingfuel_wt * sigmoidX(
            volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0
        ) + volume_wingfuel_wt * sigmoidX(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)

        J["OEM_wingfuel_mass", Mission.Design.GROSS_MASS] = dOEMwingfuelWt_dGTOW = 1
        J["OEM_wingfuel_mass", Aircraft.Propulsion.MASS] = dOEMwingfuelWt_dPropWt = -1
        J["OEM_wingfuel_mass", Aircraft.Controls.TOTAL_MASS] = dOEMwingfuelWt_dControlWt = -1
        J["OEM_wingfuel_mass", Aircraft.Design.STRUCTURE_MASS] = dOEMwingfuelWt_dStructWt = -1
        J["OEM_wingfuel_mass", Aircraft.Design.FIXED_EQUIPMENT_MASS] = dOEMwingfuelWt_dFEqWt = -1
        J["OEM_wingfuel_mass", Aircraft.Design.FIXED_USEFUL_LOAD] = dOEMwingfuelWt_dUsefulWt = -1

        J["OEM_fuel_vol", Mission.Design.GROSS_MASS] = \
            1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Propulsion.MASS] = \
            -1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Controls.TOTAL_MASS] = \
            -1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Design.STRUCTURE_MASS] = \
            -1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Design.FIXED_EQUIPMENT_MASS] = \
            -1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Design.FIXED_USEFUL_LOAD] = \
            -1 / rho_fuel * GRAV_ENGLISH_LBM
        J["OEM_fuel_vol", Aircraft.Fuel.DENSITY] = \
            -OEM_wingfuel_wt / rho_fuel**2 * GRAV_ENGLISH_LBM

        J[Aircraft.Fuel.WING_VOLUME_DESIGN, Mission.Design.FUEL_MASS_REQUIRED] = (
            1.0 + fuel_margin / 100.0) / rho_fuel * GRAV_ENGLISH_LBM
        J[Aircraft.Fuel.WING_VOLUME_DESIGN, Aircraft.Fuel.DENSITY] = (
            -(1.0 + fuel_margin / 100.0) * req_fuel_wt /
            rho_fuel**2 * GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuel.WING_VOLUME_DESIGN, Aircraft.Fuel.FUEL_MARGIN] = (
            1 / 100.0 * req_fuel_wt / rho_fuel
        )

        J["volume_wingfuel_mass", Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = rho_fuel / \
            GRAV_ENGLISH_LBM
        J["volume_wingfuel_mass", Aircraft.Fuel.DENSITY] = geometric_fuel_vol

        dMaxWFWt_dGTOW = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dGTOW
            + dOEMwingfuelWt_dGTOW *
            sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dGTOW
        )
        dMaxWFWt_dPropWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dPropWt
            + dOEMwingfuelWt_dPropWt *
            sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dPropWt
        )
        dMaxWFWt_dControlWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dControlWt
            + dOEMwingfuelWt_dControlWt
            * sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dControlWt
        )
        dMaxWFWt_dControlWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dControlWt
            + dOEMwingfuelWt_dControlWt
            * sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dControlWt
        )
        dMaxWFWt_dStructWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dStructWt
            + dOEMwingfuelWt_dStructWt
            * sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dStructWt
        )
        dMaxWFWt_dFEqWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dFEqWt
            + dOEMwingfuelWt_dFEqWt *
            sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dFEqWt
        )
        dMaxWFWt_dUsefulWt = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dUsefulWt
            + dOEMwingfuelWt_dUsefulWt
            * sigmoidX(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * dOEMwingfuelWt_dUsefulWt
        )
        dMaxWFWt_dGeomFuelVol = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * rho_fuel
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * rho_fuel
            + rho_fuel * sigmoidX(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
        )
        dMaxWFWt_dRhoFuel = (
            OEM_wingfuel_wt
            * dSigmoidXdx(volume_wingfuel_wt - OEM_wingfuel_wt, 0, 1 / 95.0)
            * geometric_fuel_vol
            + volume_wingfuel_wt
            * dSigmoidXdx(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
            * geometric_fuel_vol
            + geometric_fuel_vol *
            sigmoidX(OEM_wingfuel_wt - volume_wingfuel_wt, 0, 1 / 95.0)
        )

        J["max_wingfuel_mass", Mission.Design.GROSS_MASS] = dMaxWFWt_dGTOW
        J["max_wingfuel_mass", Aircraft.Propulsion.MASS] = dMaxWFWt_dPropWt
        J["max_wingfuel_mass", Aircraft.Controls.TOTAL_MASS] = dMaxWFWt_dControlWt
        J["max_wingfuel_mass", Aircraft.Design.STRUCTURE_MASS] = dMaxWFWt_dStructWt
        J["max_wingfuel_mass", Aircraft.Design.FIXED_EQUIPMENT_MASS] = dMaxWFWt_dFEqWt
        J["max_wingfuel_mass", Aircraft.Design.FIXED_USEFUL_LOAD] = dMaxWFWt_dUsefulWt
        J["max_wingfuel_mass", Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = dMaxWFWt_dGeomFuelVol / \
            GRAV_ENGLISH_LBM
        J["max_wingfuel_mass", Aircraft.Fuel.DENSITY] = dMaxWFWt_dRhoFuel

        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Mission.Design.GROSS_MASS] = dMaxWFWt_dGTOW / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Aircraft.Propulsion.MASS] = dMaxWFWt_dPropWt / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Aircraft.Controls.TOTAL_MASS] = dMaxWFWt_dControlWt / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Aircraft.Design.STRUCTURE_MASS] = dMaxWFWt_dStructWt / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Aircraft.Design.FIXED_EQUIPMENT_MASS] = dMaxWFWt_dFEqWt / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
            Aircraft.Design.FIXED_USEFUL_LOAD] = dMaxWFWt_dUsefulWt / (rho_fuel)
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX] = dMaxWFWt_dGeomFuelVol / (
            rho_fuel
        )
        J[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, Aircraft.Fuel.DENSITY] = dMaxWFWt_dRhoFuel / (
            rho_fuel
        ) - max_wingfuel_wt / (rho_fuel**2)


class FuelSysAndFullFuselageMass(om.ExplicitComponent):
    """
    Computation of fuselage mass and fuel system mass
    """

    def setup(self):

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')

        add_aviary_input(self, Aircraft.Wing.MASS, units='lbm')
        self.add_input("wing_mounted_mass", val=24446.343040697346, units="lbm",
                       desc="WM: mass of gear and engine (everything on wing that isn`t wing itself or fuel")
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
                         units='unitless')
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/galUS')
        add_aviary_input(self, Mission.Design.FUEL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUEL_MARGIN, units='unitless')
        self.add_input("wingfuel_mass_min", val=32850, units="lbm",
                       desc="WFWMIN: minimum wing fuel mass")

        self.add_output("fus_mass_full", val=0, units="lbm",
                        desc="WX: mass of fuselage and contents, including empennage")
        add_aviary_output(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')

        self.declare_partials(
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            [
                Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
                Mission.Design.FUEL_MASS,
                Aircraft.Fuel.FUEL_MARGIN,
            ],
        )
        self.declare_partials(
            "fus_mass_full",
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Wing.MASS,
                "wingfuel_mass_min",
                "wing_mounted_mass",
            ],
        )

    def compute(self, inputs, outputs):

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        total_wing_wt = inputs[Aircraft.Wing.MASS] * GRAV_ENGLISH_LBM
        wing_mounted_wt = inputs["wing_mounted_mass"] * GRAV_ENGLISH_LBM
        CK21 = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        c_mass_trend_fuel_sys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        fuel_wt_des = inputs[Mission.Design.FUEL_MASS] * GRAV_ENGLISH_LBM
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]
        wingfuel_wt_min = inputs["wingfuel_mass_min"] * GRAV_ENGLISH_LBM

        outputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] = (
            CK21
            * (6.687 / rho_fuel)
            * c_mass_trend_fuel_sys
            * fuel_wt_des
            * (1.0 + fuel_margin / 100.0)
            / GRAV_ENGLISH_LBM
        )
        outputs["fus_mass_full"] = (
            gross_wt_initial - total_wing_wt - wingfuel_wt_min - wing_mounted_wt
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        total_wing_wt = inputs[Aircraft.Wing.MASS] * GRAV_ENGLISH_LBM
        wing_mounted_wt = inputs["wing_mounted_mass"] * GRAV_ENGLISH_LBM
        CK21 = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        c_mass_trend_fuel_sys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        fuel_wt_des = inputs[Mission.Design.FUEL_MASS] * GRAV_ENGLISH_LBM
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]
        wingfuel_wt_min = inputs["wingfuel_mass_min"] * GRAV_ENGLISH_LBM

        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER] = (
            (6.687 / rho_fuel)
            * c_mass_trend_fuel_sys
            * fuel_wt_des
            * (1.0 + fuel_margin / 100.0)
            / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.DENSITY] = (
            -CK21
            * (6.687 / rho_fuel**2)
            * c_mass_trend_fuel_sys
            * fuel_wt_des
            * (1.0 + fuel_margin / 100.0)
        )
        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT] = (
            CK21 * (6.687 / rho_fuel) * fuel_wt_des *
            (1.0 + fuel_margin / 100.0) / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Mission.Design.FUEL_MASS] = (
            CK21
            * (6.687 / rho_fuel)
            * c_mass_trend_fuel_sys
            * (1.0 + fuel_margin / 100.0)
        )
        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.FUEL_MARGIN] = (
            CK21 * (6.687 / rho_fuel) * c_mass_trend_fuel_sys *
            fuel_wt_des * (1 / 100.0) / GRAV_ENGLISH_LBM
        )

        J["fus_mass_full", Mission.Design.GROSS_MASS] = 1
        J["fus_mass_full", Aircraft.Wing.MASS] = -1
        J["fus_mass_full", "wingfuel_mass_min"] = -1
        J["fus_mass_full", "wing_mounted_mass"] = -1


class FuselageAndStructMass(om.ExplicitComponent):
    """
    Computation of total structural group mass and mass of the fuselage structure.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        self.add_input("fus_mass_full", val=4000, units="lbm",
                       desc="WX: mass of fuselage and contents, including empennage")
        add_aviary_input(self, Aircraft.Fuselage.MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.TailBoom.LENGTH, units='ft')
        self.add_input("pylon_len", val=0, units="ft",
                       desc="ELRW: length of pylon for fuselage mounted engines")
        self.add_input("min_dive_vel", val=419.75918333, units="kn",
                       desc="VDMIN: dive velocity")
        add_aviary_input(self, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, units='psi')
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        self.add_input("MAT", val=0, units="lbm",
                       desc="WAT: Weight of the Fuselage Acoustic Treatment")
        add_aviary_input(self, Aircraft.Wing.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Wing.MASS, units='lbm')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS, units='lbm')
        add_aviary_input(self, Aircraft.VerticalTail.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.LandingGear.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Engine.POD_MASS_SCALER,
                         shape=num_engine_type, units='unitless')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURAL_MASS_INCREMENT, units='lbm')

        add_aviary_output(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

        self.declare_partials(
            Aircraft.Fuselage.MASS,
            [
                Aircraft.Fuselage.MASS_COEFFICIENT,
                "fus_mass_full",
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.TailBoom.LENGTH,
                "pylon_len",
                "min_dive_vel",
                Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                "MAT",
            ],
        )
        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, "*")

    def compute(self, inputs, outputs):

        fus_wt_full = inputs["fus_mass_full"] * GRAV_ENGLISH_LBM
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        cabin_len_tailboom = inputs[Aircraft.TailBoom.LENGTH]
        pylon_len = inputs["pylon_len"]
        min_dive_vel = inputs["min_dive_vel"]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        WAT = inputs["MAT"] * GRAV_ENGLISH_LBM
        CK8 = inputs[Aircraft.Wing.MASS_SCALER]
        total_wing_wt = inputs[Aircraft.Wing.MASS] * GRAV_ENGLISH_LBM
        CK9 = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        htail_wt = inputs[Aircraft.HorizontalTail.MASS] * GRAV_ENGLISH_LBM
        CK10 = inputs[Aircraft.VerticalTail.MASS_SCALER]
        vtail_wt = inputs[Aircraft.VerticalTail.MASS] * GRAV_ENGLISH_LBM
        CK11 = inputs[Aircraft.Fuselage.MASS_SCALER]
        CK12 = inputs[Aircraft.LandingGear.TOTAL_MASS_SCALER]
        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * \
            GRAV_ENGLISH_LBM
        CK14 = inputs[Aircraft.Engine.POD_MASS_SCALER]
        sec_wt = inputs[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS] * \
            GRAV_ENGLISH_LBM
        delta_struct_wt = inputs[Aircraft.Design.STRUCTURAL_MASS_INCREMENT] * \
            GRAV_ENGLISH_LBM

        fus_wt = (
            c_fuselage
            * (
                (fus_wt_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** 0.508
            + WAT
        )

        outputs[Aircraft.Fuselage.MASS] = fus_wt / GRAV_ENGLISH_LBM
        outputs[Aircraft.Design.STRUCTURE_MASS] = (
            CK8 * total_wing_wt
            + CK9 * htail_wt
            + CK10 * vtail_wt
            + CK11 * fus_wt
            + CK12 * landing_gear_wt
            + CK14 * sec_wt
            + delta_struct_wt
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):

        fus_wt_full = inputs["fus_mass_full"] * GRAV_ENGLISH_LBM
        c_fuselage = inputs[Aircraft.Fuselage.MASS_COEFFICIENT]
        fus_SA = inputs[Aircraft.Fuselage.WETTED_AREA]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        cabin_len_tailboom = inputs[Aircraft.TailBoom.LENGTH]
        pylon_len = inputs["pylon_len"]
        min_dive_vel = inputs["min_dive_vel"]
        p_diff_fus = inputs[Aircraft.Fuselage.PRESSURE_DIFFERENTIAL]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        WAT = inputs["MAT"] * GRAV_ENGLISH_LBM
        CK8 = inputs[Aircraft.Wing.MASS_SCALER]
        total_wing_wt = inputs[Aircraft.Wing.MASS] * GRAV_ENGLISH_LBM
        CK9 = inputs[Aircraft.HorizontalTail.MASS_SCALER]
        htail_wt = inputs[Aircraft.HorizontalTail.MASS] * GRAV_ENGLISH_LBM
        CK10 = inputs[Aircraft.VerticalTail.MASS_SCALER]
        vtail_wt = inputs[Aircraft.VerticalTail.MASS] * GRAV_ENGLISH_LBM
        CK11 = inputs[Aircraft.Fuselage.MASS_SCALER]
        CK12 = inputs[Aircraft.LandingGear.TOTAL_MASS_SCALER]
        landing_gear_wt = inputs[Aircraft.LandingGear.TOTAL_MASS] * \
            GRAV_ENGLISH_LBM
        CK14 = inputs[Aircraft.Engine.POD_MASS_SCALER]
        sec_wt = inputs[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS] * \
            GRAV_ENGLISH_LBM
        delta_struct_wt = inputs[Aircraft.Design.STRUCTURAL_MASS_INCREMENT] * \
            GRAV_ENGLISH_LBM

        x = 0.508 * (
            (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        ) ** (0.508 - 1)
        fus_wt = (
            c_fuselage
            * (
                (fus_wt_full / 10000.0) ** 0.7
                * (fus_SA / 1000.0)
                * cabin_width
                * (cabin_len_tailboom + pylon_len) ** 0.5
                * np.log10(min_dive_vel)
                * (p_diff_fus + 1.0) ** 0.2
                * ULF**0.3
            )
            ** 0.508
            + WAT
        )
        dFusWt_dCFus = (
            (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        ) ** 0.508
        dFusWt_dFusWtFull = (
            c_fuselage
            * x
            * 0.7
            * (fus_wt_full / 10000.0) ** (0.7 - 1)
            * (1 / 10000)
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dFusSA = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (1 / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dCabWidth = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dCabLenBoom = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * 0.5
            * (cabin_len_tailboom + pylon_len) ** (-0.5)
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dPylonLen = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * 0.5
            * (cabin_len_tailboom + pylon_len) ** (-0.5)
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dMinDiveVel = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * 1
            / (min_dive_vel * np.log(10))
            * (p_diff_fus + 1.0) ** 0.2
            * ULF**0.3
        )
        dFusWt_dPdiffFus = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * 0.2
            * (p_diff_fus + 1.0) ** (0.2 - 1)
            * ULF**0.3
        )
        dFusWt_dULF = (
            c_fuselage
            * x
            * (fus_wt_full / 10000.0) ** 0.7
            * (fus_SA / 1000.0)
            * cabin_width
            * (cabin_len_tailboom + pylon_len) ** 0.5
            * np.log10(min_dive_vel)
            * (p_diff_fus + 1.0) ** 0.2
            * 0.3
            * ULF ** (0.3 - 1)
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_COEFFICIENT] = \
            dFusWt_dCFus / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, "fus_mass_full"] = dFusWt_dFusWtFull
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = \
            dFusWt_dFusSA / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_DIAMETER] = \
            dFusWt_dCabWidth / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.TailBoom.LENGTH] = \
            dFusWt_dCabLenBoom / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, "pylon_len"] = \
            dFusWt_dPylonLen / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, "min_dive_vel"] = \
            dFusWt_dMinDiveVel / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = \
            dFusWt_dPdiffFus / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = \
            dFusWt_dULF / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, "MAT"] = dFusWt_dWAT = 1

        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.Fuselage.MASS_COEFFICIENT] = dFusWt_dCFus * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, "fus_mass_full"] = dFusWt_dFusWtFull * CK11
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Fuselage.WETTED_AREA] = \
            dFusWt_dFusSA * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.Fuselage.AVG_DIAMETER] = dFusWt_dCabWidth * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.TailBoom.LENGTH] = \
            dFusWt_dCabLenBoom * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, "pylon_len"] = \
            dFusWt_dPylonLen * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, "min_dive_vel"] = \
            dFusWt_dMinDiveVel * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL] = dFusWt_dPdiffFus * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = dFusWt_dULF * CK11 / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, "MAT"] = dFusWt_dWAT * CK11
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Wing.MASS_SCALER] = \
            total_wing_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Wing.MASS] = CK8
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.HorizontalTail.MASS_SCALER] = htail_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.HorizontalTail.MASS] = CK9
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.VerticalTail.MASS_SCALER] = vtail_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.VerticalTail.MASS] = CK10
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Fuselage.MASS_SCALER] = \
            fus_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS,
            Aircraft.LandingGear.TOTAL_MASS_SCALER] = landing_gear_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.LandingGear.TOTAL_MASS] = CK12
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Engine.POD_MASS_SCALER] = \
            sec_wt / GRAV_ENGLISH_LBM
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS] = CK14
        J[Aircraft.Design.STRUCTURE_MASS, Aircraft.Design.STRUCTURAL_MASS_INCREMENT] = 1


class FuelMass(om.ExplicitComponent):
    """
    Computation of fuel masses including fuel carried, total propulsion group mass,
    and minimum value of fuel mass.
    """

    def setup(self):

        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        self.add_input("eng_comb_mass", val=14371.0, units="lbm",
                       desc="WPSTAR: mass of dry engine and engine installation. Includes mass of electrical augmentation system.")
        add_aviary_input(self, Aircraft.Controls.TOTAL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.FIXED_EQUIPMENT_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.FIXED_USEFUL_LOAD, units='lbm')
        self.add_input("payload_mass_des", val=36000, units="lbm",
                       desc="WPLDES: design payload")
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
                         units='unitless')
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/galUS')
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm')

        self.add_input("payload_mass_max", val=46040, units="lbm",
                       desc="WPLMAX: maximum payload that the aircraft is being asked to carry (design payload + cargo)")

        add_aviary_input(self, Aircraft.Fuel.FUEL_MARGIN, units='unitless')

        add_aviary_output(self, Mission.Design.FUEL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_output(self, Mission.Design.FUEL_MASS_REQUIRED, units='lbm')
        self.add_output("fuel_mass_min", val=0, units="lbm",
                        desc="WFAMIN: minimum value of fuel mass (set when max payload is carried)")

        self.declare_partials(
            Mission.Design.FUEL_MASS,
            [
                Mission.Design.GROSS_MASS,
                "eng_comb_mass",
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                "payload_mass_des",
                Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
                Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.FUEL_MARGIN,
            ],
        )
        self.declare_partials(Aircraft.Propulsion.MASS, [
                              "eng_comb_mass", Aircraft.Fuel.FUEL_SYSTEM_MASS], val=1)
        self.declare_partials(Mission.Design.FUEL_MASS_REQUIRED, [
                              Mission.Design.GROSS_MASS], val=1)
        self.declare_partials(
            Mission.Design.FUEL_MASS_REQUIRED,
            [
                "eng_comb_mass",
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                Aircraft.Fuel.FUEL_SYSTEM_MASS,
                "payload_mass_des",
            ],
            val=-1,
        )
        self.declare_partials("fuel_mass_min", [Mission.Design.GROSS_MASS], val=1)
        self.declare_partials(
            "fuel_mass_min",
            [
                "eng_comb_mass",
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Controls.TOTAL_MASS,
                Aircraft.Design.FIXED_EQUIPMENT_MASS,
                Aircraft.Design.FIXED_USEFUL_LOAD,
                "payload_mass_max",
                Aircraft.Fuel.FUEL_SYSTEM_MASS,
            ],
            val=-1,
        )

    def compute(self, inputs, outputs):

        struct_wt = inputs[Aircraft.Design.STRUCTURE_MASS] * GRAV_ENGLISH_LBM
        fuel_sys_wt = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        eng_comb_wt = inputs["eng_comb_mass"] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        fixed_equip_wt = inputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] * \
            GRAV_ENGLISH_LBM
        useful_wt = inputs[Aircraft.Design.FIXED_USEFUL_LOAD] * \
            GRAV_ENGLISH_LBM
        payload_wt_des = inputs["payload_mass_des"] * GRAV_ENGLISH_LBM
        CK21 = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        c_mass_trend_fuel_sys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        payload_wt = inputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS] * \
            GRAV_ENGLISH_LBM
        payload_wt_max = inputs["payload_mass_max"] * GRAV_ENGLISH_LBM
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]

        outputs[Mission.Design.FUEL_MASS] = (
            gross_wt_initial
            - eng_comb_wt
            - struct_wt
            - control_wt
            - fixed_equip_wt
            - useful_wt
            - payload_wt_des
        ) / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        ) / GRAV_ENGLISH_LBM
        outputs[Aircraft.Propulsion.MASS] = (
            eng_comb_wt + fuel_sys_wt) / GRAV_ENGLISH_LBM
        outputs[Mission.Design.FUEL_MASS_REQUIRED] = (
            gross_wt_initial
            - eng_comb_wt
            - struct_wt
            - control_wt
            - fixed_equip_wt
            - useful_wt
            - payload_wt_des
            - fuel_sys_wt
        ) / GRAV_ENGLISH_LBM
        outputs["fuel_mass_min"] = (
            gross_wt_initial
            - eng_comb_wt
            - struct_wt
            - control_wt
            - fixed_equip_wt
            - useful_wt
            - payload_wt_max
            - fuel_sys_wt
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):

        struct_wt = inputs[Aircraft.Design.STRUCTURE_MASS] * GRAV_ENGLISH_LBM
        fuel_sys_wt = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] * GRAV_ENGLISH_LBM
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        eng_comb_wt = inputs["eng_comb_mass"] * GRAV_ENGLISH_LBM
        control_wt = inputs[Aircraft.Controls.TOTAL_MASS] * GRAV_ENGLISH_LBM
        fixed_equip_wt = inputs[Aircraft.Design.FIXED_EQUIPMENT_MASS] * \
            GRAV_ENGLISH_LBM
        useful_wt = inputs[Aircraft.Design.FIXED_USEFUL_LOAD] * \
            GRAV_ENGLISH_LBM
        payload_wt_des = inputs["payload_mass_des"] * GRAV_ENGLISH_LBM
        CK21 = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        c_mass_trend_fuel_sys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT]
        rho_fuel = inputs[Aircraft.Fuel.DENSITY] * GRAV_ENGLISH_LBM
        payload_wt = inputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS] * \
            GRAV_ENGLISH_LBM
        payload_wt_max = inputs["payload_mass_max"] * GRAV_ENGLISH_LBM
        fuel_margin = inputs[Aircraft.Fuel.FUEL_MARGIN]

        J[Mission.Design.FUEL_MASS, Mission.Design.GROSS_MASS] = 1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, "eng_comb_mass"] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Design.STRUCTURE_MASS] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Controls.TOTAL_MASS] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Design.FIXED_EQUIPMENT_MASS] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Design.FIXED_USEFUL_LOAD] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, "payload_mass_des"] = -1 / (
            1.0
            + CK21 * c_mass_trend_fuel_sys * (1 + fuel_margin / 100) * 6.687 / rho_fuel
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER] = (
            -(
                gross_wt_initial
                - eng_comb_wt
                - struct_wt
                - control_wt
                - fixed_equip_wt
                - useful_wt
                - payload_wt_des
            )
            / (
                1.0
                + CK21
                * c_mass_trend_fuel_sys
                * (1 + fuel_margin / 100)
                * 6.687
                / rho_fuel
            )
            ** 2
            * c_mass_trend_fuel_sys
            * (1 + fuel_margin / 100)
            * 6.687
            / rho_fuel / GRAV_ENGLISH_LBM
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT] = (
            -(
                gross_wt_initial
                - eng_comb_wt
                - struct_wt
                - control_wt
                - fixed_equip_wt
                - useful_wt
                - payload_wt_des
            )
            / (
                1.0
                + CK21
                * c_mass_trend_fuel_sys
                * (1 + fuel_margin / 100)
                * 6.687
                / rho_fuel
            )
            ** 2
            * CK21
            * (1 + fuel_margin / 100)
            * 6.687
            / rho_fuel / GRAV_ENGLISH_LBM
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Fuel.DENSITY] = (
            -(
                gross_wt_initial
                - eng_comb_wt
                - struct_wt
                - control_wt
                - fixed_equip_wt
                - useful_wt
                - payload_wt_des
            )
            / (
                1.0
                + CK21
                * c_mass_trend_fuel_sys
                * (1 + fuel_margin / 100)
                * 6.687
                / rho_fuel
            )
            ** 2
            * (
                -CK21
                * c_mass_trend_fuel_sys
                * (1 + fuel_margin / 100)
                * 6.687
                / rho_fuel**2
            )
        )
        J[Mission.Design.FUEL_MASS, Aircraft.Fuel.FUEL_MARGIN] = (
            -(
                gross_wt_initial
                - eng_comb_wt
                - struct_wt
                - control_wt
                - fixed_equip_wt
                - useful_wt
                - payload_wt_des
            )
            / (
                1.0
                + CK21
                * c_mass_trend_fuel_sys
                * (1 + fuel_margin / 100)
                * 6.687
                / rho_fuel
            )
            ** 2
            * CK21
            * c_mass_trend_fuel_sys
            * (1 / 100)
            * 6.687
            / rho_fuel / GRAV_ENGLISH_LBM
        )


class FuelMassGroup(om.Group):
    """
    Group of fuel related components including FuelSysAndFullFuselageMass,
    FuselageAndStructMass, FuelMass, FuelAndOEMOutputs, and BodyTankCalculations.
    """

    def setup(self):

        # variables that are calculated at a higher level
        higher_level_inputs1 = ["wing_mounted_mass"]
        higher_level_inputs2 = [
            "min_dive_vel",
        ]
        higher_level_inputs3 = [
            "payload_mass_des",
            "payload_mass_max",
            "eng_comb_mass",
        ]

        # variables that are passed within the group but not used at a higher level
        connected_inputs1 = ["wingfuel_mass_min"]
        connected_inputs2 = [
            "fus_mass_full",
        ]
        connected_inputs5 = [
            "fuel_mass_min",
            "max_wingfuel_mass",
        ]

        connected_outputs1 = ["fus_mass_full"]
        connected_outputs3 = [
            "fuel_mass_min",
        ]
        connected_outputs4 = [
            "max_wingfuel_mass",
        ]
        connected_outputs5 = ["wingfuel_mass_min"]

        self.add_subsystem(
            "sys_and_full_fus",
            FuelSysAndFullFuselageMass(),
            promotes_inputs=connected_inputs1
            + higher_level_inputs1
            + ["aircraft:*", "mission:*"],
            promotes_outputs=connected_outputs1 + ["aircraft:*"],
        )

        self.add_subsystem(
            "fus_and_struct",
            FuselageAndStructMass(),
            promotes_inputs=connected_inputs2 + higher_level_inputs2 + ["aircraft:*"],
            promotes_outputs=["aircraft:*"],
        )

        self.add_subsystem(
            "fuel",
            FuelMass(),
            promotes_inputs=higher_level_inputs3
            + ["aircraft:*", "mission:*"],
            promotes_outputs=connected_outputs3 + ["aircraft:*", "mission:*"],
        )

        self.add_subsystem(
            "fuel_and_oem",
            FuelAndOEMOutputs(),
            promotes_inputs=["aircraft:*", "mission:*"],
            promotes_outputs=connected_outputs4 + ["aircraft:*"],
        )

        self.add_subsystem(
            "body_tank",
            BodyTankCalculations(),
            promotes_inputs=connected_inputs5
            + ["aircraft:*", "mission:*"],
            promotes_outputs=connected_outputs5 + ["aircraft:*"],
        )

        self.set_input_defaults(Aircraft.Fuel.DENSITY, units="lbm/galUS")

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-9
        newton.options["rtol"] = 1e-9
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 10
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        newton.options["err_on_non_converge"] = True
        newton.options["reraise_child_analysiserror"] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1
        newton.options["err_on_non_converge"] = False

        self.linear_solver = om.DirectSolver(assemble_jac=True)
