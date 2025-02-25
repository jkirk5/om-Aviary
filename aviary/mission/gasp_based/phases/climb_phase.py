from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE


class ClimbPhaseOptions(AviaryOptionsDictionary):

    def declare_options(self):

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.'
        )

        self.declare(
            'reserve',
            types=bool,
            default=False,
            desc='Designate this phase as a reserve phase and contributes its fuel burn '
            'towards the reserve mission fuel requirements. Reserve phases should be '
            'be placed after all non-reserve phases in the phase_info.'
        )

        self.declare(
            name='target_distance',
            default=None,
            units='m',
            desc='The total distance traveled by the aircraft from takeoff to landing '
            'for the primary mission, not including reserve missions. This value must '
            'be positive.'
        )

        self.declare(
            'target_duration',
            default=None,
            units='s',
            desc='The amount of time taken by this phase added as a constraint.'
        )

        self.declare(
            name='fix_initial',
            types=bool,
            default=False,
            desc='Fixes the initial states (mass, distance) and does not allow them to '
            'change during the optimization.'
        )

        self.declare(
            name='EAS_target',
            default=0.0,
            units='kn',
            desc='Target airspeed for the balance in this phase.'
        )

        self.declare(
            name='mach_cruise',
            default=0.0,
            desc='Defines the mach constraint at the end of the phase. '
            'Only valid when target_mach=True.'
        )

        self.declare(
            'target_mach',
            types=bool,
            default=False,
            desc='Set to true to enforce a mach_constraint at the phase endpoint. '
            'The mach value is set in "mach_cruise".'
        )

        self.declare(
            name='final_altitude',
            default=0.0,
            units='ft',
            desc='Altitude for final point in the phase.'
        )

        self.declare(
            name='required_available_climb_rate',
            default=None,
            units='ft/min',
            desc='Adds a constraint requiring Dynamic.Mission.ALTITUDE_RATE_MAX to be no '
            'smaller than required_available_climb_rate. This helps to ensure that the '
            'propulsion system is large enough to handle emergency maneuvers at all points '
            'throughout the flight envelope. Default value is None for no constraint.'
        )

        self.declare(
            name='duration_bounds',
            default=(0, 0),
            units='s',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.'
        )

        self.declare(
            name='duration_ref',
            default=1.0,
            units='s',
            desc='Scale factor ref for duration.'
        )

        self.declare(
            name='alt_lower',
            types=tuple,
            default=0.0,
            units='ft',
            desc='Lower bound for altitude.'
        )

        self.declare(
            name='alt_upper',
            default=0.0,
            units='ft',
            desc='Upper bound for altitude.'
        )

        self.declare(
            name='alt_ref',
            default=1.0,
            units='ft',
            desc='Scale factor ref for altitude.'
        )

        self.declare(
            name='alt_ref0',
            default=0.0,
            units='ft',
            desc='Scale factor ref0 for altitude.'
        )

        self.declare(
            name='alt_defect_ref',
            default=None,
            units='ft',
            desc='Scale factor ref for altitude defect.'
        )

        self.declare(
            name='mass_lower',
            types=tuple,
            default=0.0,
            units='lbm',
            desc='Lower bound for mass.'
        )

        self.declare(
            name='mass_upper',
            default=0.0,
            units='lbm',
            desc='Upper bound for mass.'
        )

        self.declare(
            name='mass_ref',
            default=1.0,
            units='lbm',
            desc='Scale factor ref for mass.'
        )

        self.declare(
            name='mass_ref0',
            default=0.0,
            units='lbm',
            desc='Scale factor ref0 for mass.'
        )

        self.declare(
            name='mass_defect_ref',
            default=None,
            units='lbm',
            desc='Scale factor ref for mass defect.'
        )

        self.declare(
            name='distance_lower',
            default=0.0,
            units='NM',
            desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper',
            default=0.0,
            units='NM',
            desc='Upper bound for distance.'
        )

        self.declare(
            name='distance_ref',
            default=1.0,
            units='NM',
            desc='Scale factor ref for distance.'
        )

        self.declare(
            name='distance_ref0',
            default=0.0,
            units='NM',
            desc='Scale factor ref0 for distance.'
        )

        self.declare(
            name='distance_defect_ref',
            default=None,
            units='NM',
            desc='Scale factor ref for distance defect.'
        )

        self.declare(
            name='num_segments',
            types=int,
            default=None,
            desc='The number of segments in transcription creation in Dymos. '
        )

        self.declare(
            name='order',
            types=int,
            default=None,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos.'
        )


class ClimbPhase(PhaseBuilderBase):
    """
    A phase builder for a climb phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the climb phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the climb phase are included.
    """
    default_name = 'climb_phase'
    default_ode_class = ClimbODE
    default_options_class = ClimbPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new climb phase for analysis using these constraints.

        If ode_class is None, ClimbODE is used as the default.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = self.phase = super().build_phase(aviary_options)

        # Custom configurations for the climb phase
        user_options = self.user_options

        mach_cruise = user_options.get_val('mach_cruise')
        target_mach = user_options.get_val('target_mach')
        final_altitude = user_options.get_val('final_altitude', units='ft')
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', units='ft/min')

        # States
        self.add_altitude_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

        # Boundary Constraints
        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=final_altitude,
            units="ft",
            ref=final_altitude,
        )

        if required_available_climb_rate is not None:
            # TODO: this should be altitude rate max
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                loc="final",
                lower=required_available_climb_rate,
                units="ft/min",
                ref=1,
            )

        if target_mach:
            phase.add_boundary_constraint(
                Dynamic.Atmosphere.MACH,
                loc="final",
                equals=mach_cruise,
            )

        # Timeseries Outputs
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units="unitless",
        )
        phase.add_timeseries_output("EAS", output_name="EAS", units="kn")
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units="lbm/s"
        )
        phase.add_timeseries_output("theta", output_name="theta", units="deg")
        phase.add_timeseries_output(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            output_name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            units="deg",
        )
        phase.add_timeseries_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
            units="deg",
        )
        phase.add_timeseries_output(
            "TAS_violation", output_name="TAS_violation", units="kn")
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units="kn",
        )
        phase.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units="lbf",
        )
        phase.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

        return phase

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'EAS_target': self.user_options.get_val('EAS_target', units='kn'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
        }


ClimbPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for horizontal distance traveled')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
