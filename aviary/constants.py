from openmdao.utils import units
from aviary.variable_info.enums import FuelType

units.add_unit('distance_units', '1*m')

GRAV_EARTH = (
    9.80665,
    'm/s**2',
)  # NIST https://physics.nist.gov/cgi-bin/cuu/Value?gn|search_for=gravity

GRAV_MARS = (
    3.712,
    'm/s**2',
)  # Mars Global Reference Atmospheric Model (Mars-GRAM) 2024: User Guide, NASA/TM-20240012934

GRAV_VENUS = (
    8.870,
    'm/s**2',
)  # Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168

RADIUS_EARTH = (
    6371009.0,
    'm',
)  # Source: GRS80, mean earth radius (rounded to nearest meter)
# convert_geopotential_altitude() is a python utility function for engine decs that require RADIUS_EARTH to be specified in meters!

RADIUS_MARS = (
    3386200.0,
    'm',
)  # Mars Global Reference Atmospheric Model (Mars-GRAM) 2024: User Guide, NASA/TM-20240012934, avg of equatorial and polar radius

RADIUS_VENUS = (
    6051800.0,
    'm',
)  # Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168, avg of equatorial and polar radius

# sea level standard pressure in psf
PSLS_PSF = 2116.22

# sea level standard temperature in deg R
TSLS_DEGR = 518.67

# fuel properties
FUEL_DENSITIES = {
    # NOTE GASP and FLOPS average of 6.7 does not match average from data sheets but is in range
    # TODO update Aviary default to average from data sheet, update legacy code defaults & existing
    #      models to use FLOPS/GASP 6.7 value?
    FuelType.JET_A: (
        6.7,
        'lbm/galUS',
    ),  # 775-500 kg/m^3 @ 15 degC (source: material data sheets from major oil producers)
    FuelType.METHANE_LIQUID: (
        3.75,
        'lbm/galUS',
    ),  # 3.5-4 lbm/galUS @ -260 deg F (source: material data sheet from major gas distributor)
    FuelType.METHANE_GASEOUS: (
        0.657,
        'kg/m**3',
    ),  # 25 degC at 1 atm, a more definitive source is desired
}
