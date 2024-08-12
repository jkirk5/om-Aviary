import warnings
import math
import numpy as np
import openmdao.api as om
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Settings
from aviary.constants import RHO_SEA_LEVEL_ENGLISH, TSLS_DEGR
from aviary.utils.functions import add_aviary_input, add_aviary_output


def _unint(xa, ya, x):
    """
    univariate table routine with seperate arrays for x and y
    This routine interpolates over a 4 point interval using a
    variation of 3nd degree interpolation to produce a continuity
    of slope between adjacent intervals.
    """

    Lmt = 0
    n = len(xa)
    # test for off low end
    if xa[0] > x:
        Lmt = 1  # off low end
        y = ya[0]
    elif xa[0] == x:
        y = ya[0]  # at low end
    else:
        ifnd = 0
        idx = 0
        for i in range(1, n):
            if xa[i] == x:
                ifnd = 1  # at a node
                idx = i
                break
            elif xa[i] > x:
                ifnd = 2  # between (xa[i-1],xa[i])
                idx = i
                break
        if ifnd == 0:
            idx = n
            Lmt = 2  # off high end
            y = ya[n - 1]
        elif ifnd == 1:
            y = ya[idx]
        elif ifnd == 2:
            # jx1: the first point of four points
            if idx == 1:
                # first interval
                jx1 = 0
                ra = 1.0
            elif idx == n - 1:
                # last interval
                jx1 = n - 4
                ra = 0.0
            else:
                jx1 = idx - 2
                ra = (xa[idx] - x) / (xa[idx] - xa[idx - 1])
            rb = 1.0 - ra

            # get coefficeints and results
            p1 = xa[jx1 + 1] - xa[jx1]
            p2 = xa[jx1 + 2] - xa[jx1 + 1]
            p3 = xa[jx1 + 3] - xa[jx1 + 2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = x - xa[jx1]
            d2 = x - xa[jx1 + 1]
            d3 = x - xa[jx1 + 2]
            d4 = x - xa[jx1 + 3]
            c1 = ra / p1 * d2 / p4 * d3
            c2 = -ra / p1 * d1 / p2 * d3 + rb / p2 * d3 / p5 * d4
            c3 = ra / p2 * d1 / p4 * d2 - rb / p2 * d2 / p3 * d4
            c4 = rb / p5 * d2 / p3 * d3
            y = ya[jx1] * c1 + ya[jx1 + 1] * c2 + ya[jx1 + 2] * c3 + ya[jx1 + 3] * c4

    return y, Lmt


def _biquad(T, i, xi, yi):
    """
    This routine interpolates over a 4 point interval using a
    variation of 2nd degree interpolation to produce a continuity
    of slope between adjacent intervals.

    Table set up:
    T(i)   = table number
    T(i+1) = number of x values in xi array
    T(i+2) = number of y values in yi array
    T(i+3) = values of x in ascending order
    """

    lmt = 0
    nx = int(T[i])
    ny = int(T[i + 1])
    j1 = int(i + 2)
    j2 = j1 + nx - 1
    # search in x sense
    # jx1 = subscript of 1st x
    # search routine - input j1, j2, xi
    #                - output ra_x, rb_x, kx, jx1
    z = 0.0
    x = xi
    kx = 0
    ifnd_x = 0
    jn = 0
    xc = [0, 0, 0, 0]
    for j in range(j1, j2 + 1):
        if T[j] >= x:
            ifnd_x = 1
            jn = j
            break
    if ifnd_x == 0:
        # off high end
        x = T[j2]
        kx = 2
        # the last 4 points and curve B
        jx1 = j2 - 3
        ra_x = 0.0
    else:
        # test for -- off low end, first interval, other
        if jn < j1 + 1:
            if T[jn] != x:
                kx = 1
                x = T[j1]
        if jn <= j1 + 1:
            jx1 = j1
            ra_x = 1.0
        else:
            # test for last interval
            if j == j2:
                jx1 = j2 - 3
                ra_x = 0.0
            else:
                jx1 = jn - 2
                ra_x = (T[jn] - x) / (T[jn] - T[jn - 1])
        rb_x = 1.0 - ra_x

        # return here from search of x
        lmt = kx
        jx = jx1
        # The following code puts x values in xc blocks
        for j in range(4):
            xc[j] = T[jx1 + j]
        # get coeff. in x sense
        # coefficient routine - input x,x1,x2,x3,x4,ra_x,rb_x
        p1 = xc[1] - xc[0]
        p2 = xc[2] - xc[1]
        p3 = xc[3] - xc[2]
        p4 = p1 + p2
        p5 = p2 + p3
        d1 = x - xc[0]
        d2 = x - xc[1]
        d3 = x - xc[2]
        d4 = x - xc[3]
        cx1 = ra_x / p1 * d2 / p4 * d3
        cx2 = -ra_x / p1 * d1 / p2 * d3 + rb_x / p2 * d3 / p5 * d4
        cx3 = ra_x / p2 * d1 / p4 * d2 - rb_x / p2 * d2 / p3 * d4
        cx4 = rb_x / p5 * d2 / p3 * d3
        # return to main body

        # return here with coeff. test for univariate or bivariate
        if ny == 0:
            z = 0.0
            jy = jx + nx
            z = cx1 * T[jy] + cx2 * T[jy + 1] + cx3 * T[jy + 2] + cx4 * T[jy + 3]
        else:
            # bivariate table
            y = yi
            j3 = j2 + 1
            j4 = j3 + ny - 1
            # search in y sense
            # jy1 = subscript of 1st y
            # search routine - input j3,j4,y
            #                - output ra_y,rb_y,ky,,jy1
            ky = 0
            ifnd_y = 0
            for j in range(j3, j4 + 1):
                if T[j] >= y:
                    ifnd_y = 1
                    break
            if ifnd_y == 0:
                # off high end
                y = T[j4]
                ky = 2
                # use last 4 points and curve B
                jy1 = j4 - 3
                ra_y = 0.0
            else:
                # test for off low end, first interval
                if j < j3 + 1:
                    if T[j] != y:
                        ky = 1
                        y = T[j3]
                if j <= j3 + 1:
                    jy1 = j3
                    ra_y = 1.0
                else:
                    # test for last interval
                    if j == j4:
                        jy1 = j4 - 3
                        ra_y = 0.0
                    else:
                        jy1 = j - 2
                        ra_y = (T[j] - y) / (T[j] - T[j - 1])
            rb_y = 1.0 - ra_y

            lmt = lmt + 3 * ky
            # interpolate in y sense
            # subscript - base, num. of col., num. of y's
            jy = (j4 + 1) + (jx - i - 2) * ny + (jy1 - j3)
            yt = [0, 0, 0, 0]
            for m in range(4):
                jx = jy
                yt[m] = (
                    cx1 * T[jx]
                    + cx2 * T[jx + ny]
                    + cx3 * T[jx + 2 * ny]
                    + cx4 * T[jx + 3 * ny]
                )
                jy = jy + 1

            # the following code puts y values in yc block
            yc = [0, 0, 0, 0]
            for j in range(4):
                yc[j] = T[jy1]
                jy1 = jy1 + 1
            # get coeff. in y sense
            # coeffient routine - input y, y1, y2, y3, y4, ra_y, rb_y
            p1 = yc[1] - yc[0]
            p2 = yc[2] - yc[1]
            p3 = yc[3] - yc[2]
            p4 = p1 + p2
            p5 = p2 + p3
            d1 = y - yc[0]
            d2 = y - yc[1]
            d3 = y - yc[2]
            d4 = y - yc[3]
            cy1 = ra_y / p1 * d2 / p4 * d3
            cy2 = -ra_y / p1 * d1 / p2 * d3 + rb_y / p2 * d3 / p5 * d4
            cy3 = ra_y / p2 * d1 / p4 * d2 - rb_y / p2 * d2 / p3 * d4
            cy4 = rb_y / p5 * d2 / p3 * d3
            z = cy1 * yt[0] + cy2 * yt[1] + cy3 * yt[2] + cy4 * yt[3]

    return z, lmt


CP_Angle_table = np.array(
    [
        [  # 2 blades
            [
                0.0158,
                0.0165,
                0.0188,
                0.0230,
                0.0369,
                0.0588,
                0.0914,
                0.1340,
                0.1816,
                0.22730,
            ],  # advance_ratio = 0.0
            [
                0.0215,
                0.0459,
                0.0829,
                0.1305,
                0.1906,
                0.2554,
                0.000,
                0.000,
                0.000,
                0.0000,
            ],  # advance_ratio = 0.5
            [
                -0.0149,
                -0.0088,
                0.0173,
                0.0744,
                0.1414,
                0.2177,
                0.3011,
                0.3803,
                0.000,
                0.0000,
            ],  # advance_ratio = 1.0
            [
                -0.0670,
                -0.0385,
                0.0285,
                0.1304,
                0.2376,
                0.3536,
                0.4674,
                0.5535,
                0.000,
                0.0000,
            ],  # advance_ratio = 1.5
            [
                -0.1150,
                -0.0281,
                0.1086,
                0.2646,
                0.4213,
                0.5860,
                0.7091,
                0.000,
                0.000,
                0.0000,
            ],  # advance_ratio = 2.0
            [
                -0.1151,
                0.0070,
                0.1436,
                0.2910,
                0.4345,
                0.5744,
                0.7142,
                0.8506,
                0.9870,
                1.1175,
            ],  # advance_ratio = 3.0
            [
                -0.2427,
                0.0782,
                0.4242,
                0.7770,
                1.1164,
                1.4443,
                0.000,
                0.000,
                0.000,
                0.000,
            ],  # advance_ratio = 5.0
        ],
        [  # 4 blades
            [
                0.0311,
                0.0320,
                0.0360,
                0.0434,
                0.0691,
                0.1074,
                0.1560,
                0.2249,
                0.3108,
                0.4026,
            ],
            [0.0380, 0.0800, 0.1494, 0.2364, 0.3486, 0.4760, 0.0, 0.0, 0.0, 0.0],
            [-0.0228, -0.0109, 0.0324, 0.1326, 0.2578, 0.399, 0.5664, 0.7227, 0.0, 0.0],
            [
                -0.1252,
                -0.0661,
                0.0535,
                0.2388,
                0.4396,
                0.6554,
                0.8916,
                1.0753,
                0.0,
                0.0,
            ],
            [-0.2113, -0.0480, 0.1993, 0.4901, 0.7884, 1.099, 1.3707, 0.0, 0.0, 0.0],
            [
                -0.2077,
                0.0153,
                0.2657,
                0.5387,
                0.8107,
                1.075,
                1.3418,
                1.5989,
                1.8697,
                2.1238,
            ],
            [-0.4508, 0.1426, 0.7858, 1.448, 2.0899, 2.713, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 6 blades
            [
                0.0450,
                0.0461,
                0.0511,
                0.0602,
                0.0943,
                0.1475,
                0.2138,
                0.2969,
                0.4015,
                0.5237,
            ],
            [0.0520, 0.1063, 0.2019, 0.3230, 0.4774, 0.6607, 0.0, 0.0, 0.0, 0.0],
            [
                -0.0168,
                -0.0085,
                0.0457,
                0.1774,
                0.3520,
                0.5506,
                0.7833,
                1.0236,
                0.0,
                0.0,
            ],
            [
                -0.1678,
                -0.0840,
                0.0752,
                0.3262,
                0.6085,
                0.9127,
                1.2449,
                1.5430,
                0.0,
                0.0,
            ],
            [-0.2903, -0.0603, 0.2746, 0.6803, 1.0989, 1.5353, 1.9747, 0.0, 0.0, 0.0],
            [
                -0.2783,
                0.0259,
                0.3665,
                0.7413,
                1.1215,
                1.4923,
                1.8655,
                2.2375,
                2.6058,
                2.9831,
            ],
            [-0.6181, 0.1946, 1.0758, 1.9951, 2.8977, 3.7748, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 8 blades
            [
                0.0577,
                0.0591,
                0.0648,
                0.0751,
                0.1141,
                0.1783,
                0.2599,
                0.3551,
                0.4682,
                0.5952,
            ],
            [0.0650, 0.1277, 0.2441, 0.3947, 0.5803, 0.8063, 0.0, 0.0, 0.0, 0.0],
            [
                -0.0079,
                -0.0025,
                0.0595,
                0.2134,
                0.4266,
                0.6708,
                0.9519,
                1.2706,
                0.0,
                0.0,
            ],
            [
                -0.1894,
                -0.0908,
                0.0956,
                0.3942,
                0.7416,
                1.1207,
                1.5308,
                1.9459,
                0.0,
                0.0,
            ],
            [-0.3390, -0.0632, 0.3350, 0.8315, 1.3494, 1.890, 2.4565, 0.0, 0.0, 0.0],
            [
                -0.3267,
                0.0404,
                0.4520,
                0.9088,
                1.3783,
                1.8424,
                2.306,
                2.7782,
                3.2292,
                3.7058,
            ],
            [-0.7508, 0.2395, 1.315, 2.4469, 3.5711, 4.6638, 0.0, 0.0, 0.0, 0.0],
        ],
    ]
)
CT_Angle_table = np.array(
    [
        [  # 2 blades
            [
                0.0303,
                0.0444,
                0.0586,
                0.0743,
                0.1065,
                0.1369,
                0.1608,
                0.1767,
                0.1848,
                0.1858,
            ],
            [
                0.0205,
                0.0691,
                0.1141,
                0.1529,
                0.1785,
                0.1860,
                0.000,
                0.000,
                0.0000,
                0.0000,
            ],
            [
                -0.0976,
                -0.0566,
                0.0055,
                0.0645,
                0.1156,
                0.1589,
                0.1864,
                0.1905,
                0.000,
                0.000,
            ],
            [
                -0.1133,
                -0.0624,
                0.0111,
                0.0772,
                0.1329,
                0.1776,
                0.202,
                0.2045,
                0.000,
                0.0000,
            ],
            [
                -0.1132,
                -0.0356,
                0.0479,
                0.1161,
                0.1711,
                0.2111,
                0.2150,
                0.000,
                0.000,
                0.000,
            ],
            [
                -0.0776,
                -0.0159,
                0.0391,
                0.0868,
                0.1279,
                0.1646,
                0.1964,
                0.2213,
                0.2414,
                0.2505,
            ],
            [
                -0.1228,
                -0.0221,
                0.0633,
                0.1309,
                0.1858,
                0.2314,
                0.000,
                0.000,
                0.000,
                0.000,
            ],
        ],
        [  # 4 blades
            [
                0.0426,
                0.0633,
                0.0853,
                0.1101,
                0.1649,
                0.2204,
                0.2678,
                0.3071,
                0.3318,
                0.3416,
            ],
            [0.0318, 0.1116, 0.1909, 0.2650, 0.3241, 0.3423, 0.0, 0.0, 0.0, 0.0],
            [
                -0.1761,
                -0.0960,
                0.0083,
                0.1114,
                0.2032,
                0.2834,
                0.3487,
                0.3596,
                0.0,
                0.0,
            ],
            [-0.2155, -0.1129, 0.0188, 0.1420, 0.2401, 0.3231, 0.3850, 0.390, 0.0, 0.0],
            [-0.2137, -0.0657, 0.0859, 0.2108, 0.3141, 0.3894, 0.4095, 0.0, 0.0, 0.0],
            [
                -0.1447,
                -0.0314,
                0.0698,
                0.1577,
                0.2342,
                0.3013,
                0.3611,
                0.4067,
                0.4457,
                0.4681,
            ],
            [-0.2338, -0.0471, 0.1108, 0.2357, 0.3357, 0.4174, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 6 blades
            [
                0.0488,
                0.0732,
                0.0999,
                0.1301,
                0.2005,
                0.2731,
                0.3398,
                0.3982,
                0.4427,
                0.4648,
            ],
            [0.0375, 0.1393, 0.2448, 0.3457, 0.4356, 0.4931, 0.0, 0.0, 0.0, 0.0],
            [
                -0.2295,
                -0.1240,
                0.0087,
                0.1443,
                0.2687,
                0.3808,
                0.4739,
                0.5256,
                0.0,
                0.0,
            ],
            [
                -0.2999,
                -0.1527,
                0.0235,
                0.1853,
                0.3246,
                0.4410,
                0.5290,
                0.5467,
                0.0,
                0.0,
            ],
            [-0.3019, -0.0907, 0.1154, 0.2871, 0.429, 0.5338, 0.5954, 0.0, 0.0, 0.0],
            [
                -0.2012,
                -0.0461,
                0.0922,
                0.2125,
                0.3174,
                0.4083,
                0.4891,
                0.5549,
                0.6043,
                0.6415,
            ],
            [-0.3307, -0.0749, 0.1411, 0.3118, 0.4466, 0.5548, 0.0, 0.0, 0.0, 0.0],
        ],
        [  # 8 blades
            [
                0.0534,
                0.0795,
                0.1084,
                0.1421,
                0.2221,
                0.3054,
                0.3831,
                0.4508,
                0.5035,
                0.5392,
            ],
            [0.0423, 0.1588, 0.2841, 0.4056, 0.5157, 0.6042, 0.0, 0.0, 0.0, 0.0],
            [
                -0.2606,
                -0.1416,
                0.0097,
                0.1685,
                0.3172,
                0.4526,
                0.5655,
                0.6536,
                0.0,
                0.0,
            ],
            [
                -0.3615,
                -0.1804,
                0.0267,
                0.2193,
                0.3870,
                0.5312,
                0.6410,
                0.7032,
                0.0,
                0.0,
            ],
            [-0.3674, -0.1096, 0.1369, 0.3447, 0.5165, 0.6454, 0.7308, 0.0, 0.0, 0.0],
            [
                -0.2473,
                -0.0594,
                0.1086,
                0.2552,
                0.3830,
                0.4933,
                0.5899,
                0.6722,
                0.7302,
                0.7761,
            ],
            [-0.4165, -0.1040, 0.1597, 0.3671, 0.5289, 0.6556, 0.0, 0.0, 0.0, 0.0],
        ],
    ]
)
AFCPC = np.array(
    [
        [1.67, 1.37, 1.165, 1.0, 0.881, 0.81],
        [1.55, 1.33, 1.149, 1.0, 0.890, 0.82],
    ]
)
AFCTC = np.array(
    [
        [1.39, 1.27, 1.123, 1.0, 0.915, 0.865],
        [1.49, 1.30, 1.143, 1.0, 0.915, 0.865],
    ]
)
Act_Factor_arr = np.array([80.0, 100.0, 125.0, 150.0, 175.0, 200.0])
Blade_angle_table = np.array(
    [
        [0.0, 2.0, 4.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0],  # advance_ratio = 0.0
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 0.0, 0.0, 0.0, 0.0],  # advance_ratio = 0.5
        [
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            35.0,
            40.0,
            45.0,
            0.0,
            0.0,
        ],  # advance_ratio = 1.0
        [
            20.0,
            25.0,
            30.0,
            35.0,
            40.0,
            45.0,
            50.0,
            55.0,
            0.0,
            0.0,
        ],  # advance_ratio = 1.5
        [
            30.0,
            35.0,
            40.0,
            45.0,
            50.0,
            55.0,
            60.0,
            0.0,
            0.0,
            0.0,
        ],  # advance_ratio = 2.0
        [
            45.0,
            47.5,
            50.0,
            52.5,
            55.0,
            57.5,
            60.0,
            62.5,
            65.0,
            67.5,
        ],  # advance_ratio = 3.0
        [57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 0.0, 0.0, 0.0, 0.0],  # advance_ratio = 5.0
    ]
)
BL_P_corr_table = np.array(
    [
        [
            1.84,
            1.775,
            1.75,
            1.74,
            1.76,
            1.78,
            1.80,
            1.81,
            1.835,
            1.85,
            1.865,
            1.875,
            1.88,
            1.88,
        ],  # 2 blades
        [
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.000,
            1.000,
            1.000,
        ],  # 4 blades
        [
            0.585,
            0.635,
            0.675,
            0.710,
            0.738,
            0.745,
            0.758,
            0.755,
            0.705,
            0.735,
            0.710,
            0.7250,
            0.7250,
            0.7250,
        ],  # 6 blades
        [
            0.415,
            0.460,
            0.505,
            0.535,
            0.560,
            0.575,
            0.600,
            0.610,
            0.630,
            0.630,
            0.610,
            0.6050,
            0.6000,
            0.6000,
        ],  # 8 blades
    ]
)
BL_T_corr_table = np.array(
    [
        [
            1.58,
            1.685,
            1.73,
            1.758,
            1.777,
            1.802,
            1.828,
            1.839,
            1.848,
            1.850,
            1.850,
            1.850,
            1.850,
            1.850,
        ],  # 2 blades
        [
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
        ],  # 4 blades
        [
            0.918,
            0.874,
            0.844,
            0.821,
            0.802,
            0.781,
            0.764,
            0.752,
            0.750,
            0.750,
            0.750,
            0.750,
            0.750,
            0.750,
        ],  # 6 blades
        [
            0.864,
            0.797,
            0.758,
            0.728,
            0.701,
            0.677,
            0.652,
            0.640,
            0.630,
            0.622,
            0.620,
            0.620,
            0.620,
            0.620,
        ],  # 8 blades
    ]
)
CL_arr = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
CP_CLi_table = np.array(
    [
        [
            0.0114,
            0.0294,
            0.0491,
            0.0698,
            0.0913,
            0.1486,
            0.2110,
            0.2802,
            0.3589,
            0.4443,
            0.5368,
            0.6255,
            0.00,
            0.00,
            0.00,
        ],  # CLI = 0.3
        [
            0.016,
            0.020,
            0.0294,
            0.0478,
            0.0678,
            0.0893,
            0.1118,
            0.1702,
            0.2335,
            0.3018,
            0.3775,
            0.4610,
            0.5505,
            0.6331,
            0.00,
        ],  # CLI = 0.4
        [
            0.00,
            0.0324,
            0.0486,
            0.0671,
            0.0875,
            0.1094,
            0.1326,
            0.1935,
            0.2576,
            0.3259,
            0.3990,
            0.4805,
            0.5664,
            0.6438,
            0.00,
        ],  # CLI = 0.5
        [
            0.00,
            0.029,
            0.043,
            0.048,
            0.049,
            0.0524,
            0.0684,
            0.0868,
            0.1074,
            0.1298,
            0.1537,
            0.2169,
            0.3512,
            0.5025,
            0.6605,
        ],  # CLI = 0.6
        [
            0.00,
            0.0510,
            0.0743,
            0.0891,
            0.1074,
            0.1281,
            0.1509,
            0.1753,
            0.2407,
            0.3083,
            0.3775,
            0.4496,
            0.5265,
            0.6065,
            0.6826,
        ],  # CLI = 0.7
        [
            0.00,
            0.0670,
            0.0973,
            0.1114,
            0.1290,
            0.1494,
            0.1723,
            0.1972,
            0.2646,
            0.3345,
            0.4047,
            0.4772,
            0.5532,
            0.6307,
            0.7092,
        ],  # CLI = 0.8
    ]
)
CPEC = np.array(
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
)
CT_CLi_table = np.array(
    [
        [
            0.0013,
            0.0211,
            0.0407,
            0.0600,
            0.0789,
            0.1251,
            0.1702,
            0.2117,
            0.2501,
            0.2840,
            0.3148,
            0.3316,
            0.00,
            0.00,
        ],  # CLI = 0.3
        [
            0.005,
            0.010,
            0.0158,
            0.0362,
            0.0563,
            0.0761,
            0.0954,
            0.1419,
            0.1868,
            0.2278,
            0.2669,
            0.3013,
            0.3317,
            0.3460,
        ],  # CLI = 0.4
        [
            0.00,
            0.0083,
            0.0297,
            0.0507,
            0.0713,
            0.0916,
            0.1114,
            0.1585,
            0.2032,
            0.2456,
            0.2834,
            0.3191,
            0.3487,
            0.3626,
        ],  # CLI = 0.5
        [
            0.0130,
            0.0208,
            0.0428,
            0.0645,
            0.0857,
            0.1064,
            0.1267,
            0.1748,
            0.2195,
            0.2619,
            0.2995,
            0.3350,
            0.3647,
            0.3802,
        ],  # CLI = 0.6
        [
            0.026,
            0.0331,
            0.0552,
            0.0776,
            0.0994,
            0.1207,
            0.1415,
            0.1907,
            0.2357,
            0.2778,
            0.3156,
            0.3505,
            0.3808,
            0.3990,
        ],  # CLI = 0.7
        [
            0.0365,
            0.0449,
            0.0672,
            0.0899,
            0.1125,
            0.1344,
            0.1556,
            0.2061,
            0.2517,
            0.2937,
            0.3315,
            0.3656,
            0.3963,
            0.4186,
        ],  # CLI = 0.8
    ]
)
CTEC = np.array(
    [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44]
)
# array length for CP_Angle_table and CT_Angle_table
ang_arr_len = np.array([10, 6, 8, 8, 7, 10, 6])
# array length for CP_CLi_table and CT_CLi_table
cli_arr_len = np.array([12, 14, 14, 15, 15, 15])
# integrated design lift coefficient adjustment factor to power coefficient
PF_CLI_arr = np.array([1.68, 1.405, 1.0, 0.655, 0.442, 0.255, 0.102])
# integrated design lift coefficient adjustment factor to thrust coefficient
TF_CLI_arr = np.array([1.22, 1.105, 1.0, 0.882, 0.792, 0.665, 0.540])
num_blades_arr = np.array([2.0, 4.0, 6.0, 8.0])
XPCLI = np.array(
    [
        [
            4.26,
            2.285,
            1.780,
            1.568,
            1.452,
            1.300,
            1.220,
            1.160,
            1.110,
            1.085,
            1.054,
            1.048,
            0.000,
            0.000,
            0.0,
        ],  # CL = 0.3
        [
            2.0,
            1.88,
            1.652,
            1.408,
            1.292,
            1.228,
            1.188,
            1.132,
            1.105,
            1.08,
            1.058,
            1.042,
            1.029,
            1.0220,
            0.0,
        ],  # CL = 0.4
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            0.0,
        ],  # CL = 0.5
        [
            0.0,
            0.065,
            0.40,
            0.52,
            0.551,
            0.619,
            0.712,
            0.775,
            0.815,
            0.845,
            0.865,
            0.891,
            0.928,
            0.958,
            0.975,
        ],  # CL = 0.6
        [
            0.00,
            0.250,
            0.436,
            0.545,
            0.625,
            0.682,
            0.726,
            0.755,
            0.804,
            0.835,
            0.864,
            0.889,
            0.914,
            0.935,
            0.944,
        ],  # CL = 0.7
        [
            0.00,
            0.110,
            0.333,
            0.436,
            0.520,
            0.585,
            0.635,
            0.670,
            0.730,
            0.770,
            0.807,
            0.835,
            0.871,
            0.897,
            0.909,
        ],  # CL = 0.8
    ]
)
XTCLI = np.array(
    [
        [
            22.85,
            2.40,
            1.75,
            1.529,
            1.412,
            1.268,
            1.191,
            1.158,
            1.130,
            1.122,
            1.108,
            1.108,
            0.000,
            0.000,
        ],  # CL = 0.3
        [
            5.5,
            2.27,
            1.880,
            1.40,
            1.268,
            1.208,
            1.170,
            1.110,
            1.089,
            1.071,
            1.060,
            1.054,
            1.051,
            1.048,
        ],  # CL = 0.4
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
            1.000,
        ],  # CL = 0.5
        [
            0.295,
            0.399,
            0.694,
            0.787,
            0.831,
            0.860,
            0.881,
            0.908,
            0.926,
            0.940,
            0.945,
            0.951,
            0.958,
            0.958,
        ],  # CL = 0.6
        [
            0.166,
            0.251,
            0.539,
            0.654,
            0.719,
            0.760,
            0.788,
            0.831,
            0.865,
            0.885,
            0.900,
            0.910,
            0.916,
            0.916,
        ],  # CL = 0.7
        [
            0.042,
            0.1852,
            0.442,
            0.565,
            0.635,
            0.681,
            0.716,
            0.769,
            0.809,
            0.838,
            0.855,
            0.874,
            0.881,
            0.881,
        ],  # CL = 0.8
    ]
)
advance_ratio_array2 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
advance_ratio_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
mach_tip_corr_arr = np.array([0.928, 0.916, 0.901, 0.884, 0.865, 0.845])
mach_corr_table = np.array(
    [  # ZMCRL
        [
            0.0,
            0.151,
            0.299,
            0.415,
            0.505,
            0.578,
            0.620,
            0.630,
            0.630,
            0.630,
            0.630,
        ],  # CL = 0.3
        [
            0.0,
            0.146,
            0.287,
            0.400,
            0.487,
            0.556,
            0.595,
            0.605,
            0.605,
            0.605,
            0.605,
        ],  # CL = 0.4
        [
            0.0,
            0.140,
            0.276,
            0.387,
            0.469,
            0.534,
            0.571,
            0.579,
            0.579,
            0.579,
            0.579,
        ],  # CL = 0.5
        [
            0.0,
            0.135,
            0.265,
            0.372,
            0.452,
            0.512,
            0.547,
            0.554,
            0.554,
            0.554,
            0.554,
        ],  # CL = 0.6
        [
            0.0,
            0.130,
            0.252,
            0.357,
            0.434,
            0.490,
            0.522,
            0.526,
            0.526,
            0.526,
            0.526,
        ],  # CL = 0.7
        [
            0.0,
            0.125,
            0.240,
            0.339,
            0.416,
            0.469,
            0.498,
            0.500,
            0.500,
            0.500,
            0.500,
        ],  # CL = 0.8
    ]
)
comp_mach_CT_arr = np.array(
    [
        # table number, number of X array, number of Y array, X array
        1,
        9,
        12,
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        # Y array (CTE)
        0.01,
        0.02,
        0.04,
        0.08,
        0.12,
        0.16,
        0.20,
        0.24,
        0.28,
        0.32,
        0.36,
        0.40,
        # Mach
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,  # X = 0.00
        0.979,
        0.981,
        0.984,
        0.987,
        0.990,
        0.993,
        0.996,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,  # X = 0.02
        0.944,
        0.945,
        0.950,
        0.958,
        0.966,
        0.975,
        0.984,
        0.990,
        0.996,
        0.999,
        1.00,
        1.00,  # X = 0.04
        0.901,
        0.905,
        0.912,
        0.927,
        0.942,
        0.954,
        0.964,
        0.974,
        0.984,
        0.990,
        0.900,
        0.900,  # X = 0.06
        0.862,
        0.866,
        0.875,
        0.892,
        0.909,
        0.926,
        0.942,
        0.957,
        0.970,
        0.980,
        0.984,
        0.984,  # X = 0.08
        0.806,
        0.813,
        0.825,
        0.851,
        0.877,
        0.904,
        0.924,
        0.939,
        0.952,
        0.961,
        0.971,
        0.976,  # X = 0.10
        0.675,
        0.685,
        0.700,
        0.735,
        0.777,
        0.810,
        0.845,
        0.870,
        0.890,
        0.905,
        0.920,
        0.930,  # X = 0.15
        0.525,
        0.540,
        0.565,
        0.615,
        0.670,
        0.710,
        0.745,
        0.790,
        0.825,
        0.860,
        0.880,
        0.895,  # X = 0.20
        0.225,
        0.260,
        0.320,
        0.375,
        0.430,
        0.495,
        0.550,
        0.610,
        0.660,
        0.710,
        0.740,
        0.775,  # X = 0.30
    ]
)


class PreHamiltonStandard(om.ExplicitComponent):
    """
    Pre-process parameters needed by HamiltonStandard component
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Engine.PROPELLER_DIAMETER, val=0.0, units='ft')
        add_aviary_input(
            self, Dynamic.Mission.PROPELLER_TIP_SPEED, val=np.zeros(nn), units='ft/s'
        )
        add_aviary_input(
            self, Dynamic.Mission.SHAFT_POWER, val=np.zeros(nn), units='hp'
        )
        add_aviary_input(
            self, Dynamic.Mission.DENSITY, val=np.zeros(nn), units='slug/ft**3'
        )
        add_aviary_input(self, Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='ft/s')
        add_aviary_input(
            self, Dynamic.Mission.SPEED_OF_SOUND, val=np.zeros(nn), units='ft/s'
        )

        self.add_output('power_coefficient', val=np.zeros(nn), units='unitless')
        self.add_output('advance_ratio', val=np.zeros(nn), units='unitless')
        self.add_output('tip_mach', val=np.zeros(nn), units='unitless')
        self.add_output('density_ratio', val=np.zeros(nn), units='unitless')

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'density_ratio', Dynamic.Mission.DENSITY, rows=arange, cols=arange
        )
        self.declare_partials(
            'tip_mach',
            [
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                Dynamic.Mission.SPEED_OF_SOUND,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'advance_ratio',
            [
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.PROPELLER_TIP_SPEED,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'power_coefficient',
            [
                Dynamic.Mission.SHAFT_POWER,
                Dynamic.Mission.DENSITY,
                Dynamic.Mission.PROPELLER_TIP_SPEED,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials('power_coefficient', Aircraft.Engine.PROPELLER_DIAMETER)

    def compute(self, inputs, outputs):
        diam_prop = inputs[Aircraft.Engine.PROPELLER_DIAMETER]
        shp = inputs[Dynamic.Mission.SHAFT_POWER]
        vtas = inputs[Dynamic.Mission.VELOCITY]
        tipspd = inputs[Dynamic.Mission.PROPELLER_TIP_SPEED]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        # arbitrarily small number to keep advance ratio nonzero, which allows for static thrust prediction
        # NOTE need for a separate static thrust calc method?
        vktas[np.where(vktas <= 1e-6)] = 1e-6
        density_ratio = inputs[Dynamic.Mission.DENSITY] / RHO_SEA_LEVEL_ENGLISH

        if diam_prop <= 0.0:
            raise om.AnalysisError(
                "Aircraft.Engine.PROPELLER_DIAMETER must be positive."
            )
        if any(tipspd) <= 0.0:
            raise om.AnalysisError(
                "Dynamic.Mission.PROPELLER_TIP_SPEED must be positive."
            )
        if any(sos) <= 0.0:
            raise om.AnalysisError("Dynamic.Mission.SPEED_OF_SOUND must be positive.")
        if any(density_ratio) <= 0.0:
            raise om.AnalysisError("Dynamic.Mission.DENSITY must be positive.")
        if any(shp) < 0.0:
            raise om.AnalysisError("Dynamic.Mission.SHAFT_POWER must be non-negative.")

        outputs['density_ratio'] = density_ratio
        # 1118.21948771 is speed of sound at sea level
        # TODO tip mach was already calculated, revisit this
        outputs['tip_mach'] = tipspd / sos
        outputs['advance_ratio'] = math.pi * vtas / tipspd
        # TODO back out what is going on with unit conversion factor 10e10/(2*6966)
        outputs['power_coefficient'] = (
            shp * 10.0e10 / (2 * 6966.0) / density_ratio / (tipspd**3 * diam_prop**2)
        )

    def compute_partials(self, inputs, partials):
        vtas = inputs[Dynamic.Mission.VELOCITY]
        tipspd = inputs[Dynamic.Mission.PROPELLER_TIP_SPEED]
        rho = inputs[Dynamic.Mission.DENSITY]
        diam_prop = inputs[Aircraft.Engine.PROPELLER_DIAMETER]
        shp = inputs[Dynamic.Mission.SHAFT_POWER]
        sos = inputs[Dynamic.Mission.SPEED_OF_SOUND]

        unit_conversion_const = 10.0e10 / (2 * 6966.0)

        partials["density_ratio", Dynamic.Mission.DENSITY] = 1 / RHO_SEA_LEVEL_ENGLISH
        partials["tip_mach", Dynamic.Mission.PROPELLER_TIP_SPEED] = 1 / sos
        partials["tip_mach", Dynamic.Mission.SPEED_OF_SOUND] = -tipspd / sos**2
        partials["advance_ratio", Dynamic.Mission.VELOCITY] = math.pi / tipspd
        partials["advance_ratio", Dynamic.Mission.PROPELLER_TIP_SPEED] = (
            -math.pi * vtas / (tipspd * tipspd)
        )
        partials["power_coefficient", Dynamic.Mission.SHAFT_POWER] = (
            unit_conversion_const
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * tipspd**3 * diam_prop**2)
        )
        partials["power_coefficient", Dynamic.Mission.DENSITY] = (
            -unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * rho * tipspd**3 * diam_prop**2)
        )
        partials["power_coefficient", Dynamic.Mission.PROPELLER_TIP_SPEED] = (
            -3
            * unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * tipspd**4 * diam_prop**2)
        )
        partials["power_coefficient", Aircraft.Engine.PROPELLER_DIAMETER] = (
            -2
            * unit_conversion_const
            * shp
            * RHO_SEA_LEVEL_ENGLISH
            / (rho * tipspd**3 * diam_prop**3)
        )


class HamiltonStandard(om.ExplicitComponent):
    """
    This is Hamilton Standard component rewritten from Fortran code.
    The original documentation is available at
    https://ntrs.nasa.gov/api/citations/19720010354/downloads/19720010354.pdf
    It computes the thrust coefficient of a propeller blade.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('power_coefficient', val=np.zeros(nn), units='unitless')
        self.add_input('advance_ratio', val=np.zeros(nn), units='unitless')
        add_aviary_input(self, Dynamic.Mission.MACH, val=np.zeros(nn), units='unitless')
        self.add_input('tip_mach', val=np.zeros(nn), units='unitless')
        add_aviary_input(
            self, Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, val=0.0, units='unitless'
        )  # Actitivty Factor per Blade
        add_aviary_input(
            self,
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
            val=0.0,
            units='unitless',
        )  # blade integrated lift coeff

        self.add_output('thrust_coefficient', val=np.zeros(nn), units='unitless')
        # propeller tip compressibility loss factor
        self.add_output('comp_tip_loss_factor', val=np.zeros(nn), units='unitless')

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        verbosity = self.options['aviary_options'].get_val(Settings.VERBOSITY)
        num_blades = self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_PROPELLER_BLADES
        )

        for i_node in range(self.options['num_nodes']):
            ichck = 0
            run_flag = 0
            xft = 1.0
            AF_adj_CP = np.zeros(7)  # AFCP: an AF adjustment of CP to be assigned
            AF_adj_CT = np.zeros(7)  # AFCT: an AF adjustment of CT to be assigned
            CTT = np.zeros(7)
            BLL = np.zeros(7)
            BLLL = np.zeros(7)
            PXCLI = np.zeros(7)
            XFFT = np.zeros(6)
            CTG = np.zeros(11)
            CTG1 = np.zeros(11)
            TXCLI = np.zeros(6)
            CTTT = np.zeros(4)
            XXXFT = np.zeros(4)
            act_factor = inputs[Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR]
            for k in range(2):
                AF_adj_CP[k], run_flag = _unint(Act_Factor_arr, AFCPC[k], act_factor)
                AF_adj_CT[k], run_flag = _unint(Act_Factor_arr, AFCTC[k], act_factor)
            for k in range(2, 7):
                AF_adj_CP[k] = AF_adj_CP[1]
                AF_adj_CT[k] = AF_adj_CT[1]
            if inputs['advance_ratio'][i_node] <= 0.5:
                AFCTE = (
                    2.0
                    * inputs['advance_ratio'][i_node]
                    * (AF_adj_CT[1] - AF_adj_CT[0])
                    + AF_adj_CT[0]
                )
            else:
                AFCTE = AF_adj_CT[1]

            # bounding J (advance ratio) for setting up interpolation
            if inputs['advance_ratio'][i_node] <= 1.0:
                J_begin = 0
                J_end = 3
            elif inputs['advance_ratio'][i_node] <= 1.5:
                J_begin = 1
                J_end = 4
            elif inputs['advance_ratio'][i_node] <= 2.0:
                J_begin = 2
                J_end = 5
            else:
                J_begin = 3
                J_end = 6

            CL_tab_idx_begin = 0  # NCLT
            CL_tab_idx_end = 0  # NCLTT
            # flag that given lift coeff (cli) does not fall on a node point of CL_arr
            CL_tab_idx_flg = 0  # NCL_flg
            ifnd = 0
            cli = inputs[Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT]
            power_coefficient = inputs['power_coefficient'][i_node]
            for ii in range(6):
                cl_idx = ii
                if abs(cli - CL_arr[ii]) <= 0.0009:
                    ifnd = 1
                    break
            if ifnd == 0:
                if cli <= 0.6:
                    CL_tab_idx_begin = 0
                    CL_tab_idx_end = 3
                elif cli <= 0.7:
                    CL_tab_idx_begin = 1
                    CL_tab_idx_end = 4
                else:
                    CL_tab_idx_begin = 2
                    CL_tab_idx_end = 5
            else:
                CL_tab_idx_begin = cl_idx
                CL_tab_idx_end = cl_idx
                # flag that given lift coeff (cli) falls on a node point of CL_arr
                CL_tab_idx_flg = 1

            lmod = (num_blades % 2) + 1
            if lmod == 1:
                nbb = 1
                idx_blade = int(num_blades / 2.0)
                # even number of blades idx_blade = 1 if 2 blades;
                #                       idx_blade = 2 if 4 blades;
                #                       idx_blade = 3 if 6 blades;
                #                       idx_blade = 4 if 8 blades.
                idx_blade = idx_blade - 1
            else:
                nbb = 4
                # odd number of blades
                idx_blade = 0  # start from first blade

            for ibb in range(nbb):
                # nbb = 1 even number of blades. No interpolation needed
                # nbb = 4 odd number of blades. So, interpolation done
                #       using 4 sets of even J (advance ratio) interpolation
                for kdx in range(J_begin, J_end + 1):
                    CP_Eff = power_coefficient * AF_adj_CP[kdx]
                    PBL, run_flag = _unint(CPEC, BL_P_corr_table[idx_blade], CP_Eff)
                    # PBL = number of blades correction for power_coefficient
                    CPE1 = CP_Eff * PBL * PF_CLI_arr[kdx]
                    CL_tab_idx = CL_tab_idx_begin
                    for kl in range(CL_tab_idx_begin, CL_tab_idx_end + 1):
                        CPE1X = CPE1
                        if CPE1 < CP_CLi_table[CL_tab_idx][0]:
                            CPE1X = CP_CLi_table[CL_tab_idx][0]
                        cli_len = cli_arr_len[CL_tab_idx]
                        PXCLI[kl], run_flag = _unint(
                            CP_CLi_table[CL_tab_idx][:cli_len], XPCLI[CL_tab_idx], CPE1X
                        )
                        if run_flag == 1:
                            ichck = ichck + 1
                        if verbosity is Verbosity.DEBUG or ichck <= 1:
                            if run_flag == 1:
                                warnings.warn(
                                    f"Mach,VTMACH,J,power_coefficient,CP_Eff =: {inputs[Dynamic.Mission.MACH][i_node]},{inputs['tip_mach'][i_node]},{inputs['advance_ratio'][i_node]},{power_coefficient},{CP_Eff}"
                                )
                            if kl == 4 and CPE1 < 0.010:
                                print(
                                    f"Extrapolated data is being used for CLI=.6--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5"
                                )
                            if kl == 5 and CPE1 < 0.010:
                                print(
                                    f"Extrapolated data is being used for CLI=.7--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5"
                                )
                            if kl == 6 and CPE1 < 0.010:
                                print(
                                    f"Extrapolated data is being used for CLI=.8--CPE1,PXCLI,L= , {CPE1},{PXCLI[kl]},{idx_blade}   Suggest inputting CLI=.5"
                                )
                        NERPT = 1
                        CL_tab_idx = CL_tab_idx + 1
                    if CL_tab_idx_flg != 1:
                        PCLI, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            PXCLI[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            inputs[
                                Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT
                            ],
                        )
                    else:
                        PCLI = PXCLI[CL_tab_idx_begin]
                        # PCLI = CLI adjustment to power_coefficient
                    CP_Eff = CP_Eff * PCLI  # the effective CP at baseline point for kdx
                    ang_len = ang_arr_len[kdx]
                    BLL[kdx], run_flag = _unint(
                        CP_Angle_table[idx_blade][kdx][:ang_len],
                        Blade_angle_table[kdx],
                        CP_Eff,
                    )  # blade angle at baseline point for kdx
                    try:
                        CTT[kdx], run_flag = _unint(
                            Blade_angle_table[kdx],
                            CT_Angle_table[idx_blade][kdx][:ang_len],
                            BLL[kdx],
                        )  # thrust coeff at baseline point for kdx
                    except IndexError:
                        raise om.AnalysisError(
                            "interp failed for CTT (thrust coefficient) in hamilton_standard.py"
                        )
                    if run_flag > 1:
                        NERPT = 2
                        print(
                            f"ERROR IN PROP. PERF.-- NERPT={NERPT}, run_flag={run_flag}"
                        )

                BLLL[ibb], run_flag = _unint(
                    advance_ratio_array[J_begin: J_begin + 4],
                    BLL[J_begin: J_begin + 4],
                    inputs['advance_ratio'][i_node],
                )
                ang_blade = BLLL[ibb]
                CTTT[ibb], run_flag = _unint(
                    advance_ratio_array[J_begin: J_begin + 4],
                    CTT[J_begin: J_begin + 4],
                    inputs['advance_ratio'][i_node],
                )

                # make extra correction. CTG is an "error" function, and the iteration (loop counter = "IL") tries to drive CTG/CT to 0
                # ERR_CT = CTG1[il]/CTTT[ibb], where CTG1 =CT_Eff - CTTT(IBB).
                CTG[0] = 0.100
                CTG[1] = 0.200
                TFCLII, run_flag = _unint(
                    advance_ratio_array, TF_CLI_arr, inputs['advance_ratio'][i_node]
                )
                NCTG = 10
                ifnd1 = 0
                ifnd2 = 0
                for il in range(NCTG):
                    ct = CTG[il]
                    CT_Eff = CTG[il] * AFCTE
                    TBL, run_flag = _unint(CTEC, BL_T_corr_table[idx_blade], CT_Eff)
                    # TBL = number of blades correction for thrust_coefficient
                    CTE1 = CT_Eff * TBL * TFCLII
                    CL_tab_idx = CL_tab_idx_begin
                    for kl in range(CL_tab_idx_begin, CL_tab_idx_end + 1):
                        CTE1X = CTE1
                        if CTE1 < CT_CLi_table[CL_tab_idx][0]:
                            CTE1X = CT_CLi_table[CL_tab_idx][0]
                        cli_len = cli_arr_len[CL_tab_idx]
                        TXCLI[kl], run_flag = _unint(
                            CT_CLi_table[CL_tab_idx][:cli_len],
                            XTCLI[CL_tab_idx][:cli_len],
                            CTE1X,
                        )
                        NERPT = 5
                        if run_flag == 1:
                            # off lower bound only.
                            print(
                                f"ERROR IN PROP. PERF.-- NERPT={NERPT}, run_flag={run_flag}, il = {il}, kl = {kl}"
                            )
                        if inputs['advance_ratio'][i_node] != 0.0:
                            ZMCRT, run_flag = _unint(
                                advance_ratio_array2,
                                mach_corr_table[CL_tab_idx],
                                inputs['advance_ratio'][i_node],
                            )
                            DMN = inputs[Dynamic.Mission.MACH][i_node] - ZMCRT
                        else:
                            ZMCRT = mach_tip_corr_arr[CL_tab_idx]
                            DMN = inputs['tip_mach'][i_node] - ZMCRT
                        XFFT[kl] = 1.0  # compressibility tip loss factor
                        if DMN > 0.0:
                            CTE2 = CT_Eff * TXCLI[kl] * TBL
                            XFFT[kl], run_flag = _biquad(comp_mach_CT_arr, 1, DMN, CTE2)
                        CL_tab_idx = CL_tab_idx + 1
                    if CL_tab_idx_flg != 1:
                        cli = inputs[
                            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT
                        ]
                        TCLII, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            TXCLI[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            cli,
                        )
                        xft, run_flag = _unint(
                            CL_arr[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            XFFT[CL_tab_idx_begin: CL_tab_idx_begin + 4],
                            cli,
                        )
                    else:
                        TCLII = TXCLI[CL_tab_idx_begin]
                        xft = XFFT[CL_tab_idx_begin]
                    ct = CTG[il]
                    CT_Eff = CTG[il] * AFCTE * TCLII
                    CTG1[il] = CT_Eff - CTTT[ibb]
                    if abs(CTG1[il] / CTTT[ibb]) < 0.001:
                        ifnd1 = 1
                        break
                    if il > 0:
                        CTG[il + 1] = (
                            -CTG1[il - 1]
                            * (CTG[il] - CTG[il - 1])
                            / (CTG1[il] - CTG1[il - 1])
                            + CTG[il - 1]
                        )
                        if CTG[il + 1] <= 0:
                            ifnd2 = 1
                            break

                if ifnd1 == 0 and ifnd2 == 0:
                    raise ValueError(
                        "Integrated design cl adjustment not working properly for ct "
                        f"definition (ibb={ibb})"
                    )
                if ifnd1 == 0 and ifnd2 == 1:
                    ct = 0.0
                CTTT[ibb] = ct
                XXXFT[ibb] = xft
                idx_blade = idx_blade + 1

            if nbb != 1:
                # interpolation by the number of blades if odd number
                ang_blade, run_flag = _unint(num_blades_arr, BLLL[:4], num_blades)
                ct, run_flag = _unint(num_blades_arr, CTTT, num_blades)
                xft, run_flag = _unint(num_blades_arr, XXXFT, num_blades)

            # NOTE this could be handled via the metamodel comps (extrapolate flag)
            if ichck > 0:
                print(
                    f"  table look-up error = {ichck} (if you go outside the tables.)"
                )

            outputs['thrust_coefficient'][i_node] = ct
            outputs['comp_tip_loss_factor'][i_node] = xft


class PostHamiltonStandard(om.ExplicitComponent):
    """
    Post-process after HamiltonStandard run to get thrust and compressibility
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Engine.PROPELLER_DIAMETER, val=0.0, units='ft')
        self.add_input('install_loss_factor', val=np.zeros(nn), units='unitless')
        self.add_input('thrust_coefficient', val=np.zeros(nn), units='unitless')
        self.add_input('comp_tip_loss_factor', val=np.zeros(nn), units='unitless')
        add_aviary_input(
            self, Dynamic.Mission.PROPELLER_TIP_SPEED, val=np.zeros(nn), units='ft/s'
        )
        self.add_input('density_ratio', val=np.zeros(nn), units='unitless')
        self.add_input('advance_ratio', val=np.zeros(nn), units='unitless')
        self.add_input('power_coefficient', val=np.zeros(nn), units='unitless')

        self.add_output(
            'thrust_coefficient_comp_loss', val=np.zeros(nn), units='unitless'
        )
        add_aviary_output(self, Dynamic.Mission.THRUST, val=np.zeros(nn), units='lbf')
        # keep them for reporting but don't seem to be required
        self.add_output('propeller_efficiency', val=np.zeros(nn), units='unitless')
        self.add_output('install_efficiency', val=np.zeros(nn), units='unitless')

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(
            'thrust_coefficient_comp_loss',
            [
                'thrust_coefficient',
                'comp_tip_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Mission.THRUST,
            [
                'thrust_coefficient',
                'comp_tip_loss_factor',
                Dynamic.Mission.PROPELLER_TIP_SPEED,
                'density_ratio',
                'install_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Mission.THRUST,
            [
                Aircraft.Engine.PROPELLER_DIAMETER,
            ],
        )
        self.declare_partials(
            'propeller_efficiency',
            [
                'advance_ratio',
                'power_coefficient',
                'thrust_coefficient',
                'comp_tip_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'install_efficiency',
            [
                'advance_ratio',
                'power_coefficient',
                'thrust_coefficient',
                'comp_tip_loss_factor',
                'install_loss_factor',
            ],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        ctx = inputs['thrust_coefficient'] * inputs['comp_tip_loss_factor']
        outputs['thrust_coefficient_comp_loss'] = ctx
        diam_prop = inputs[Aircraft.Engine.PROPELLER_DIAMETER]
        tipspd = inputs[Dynamic.Mission.PROPELLER_TIP_SPEED]
        install_loss_factor = inputs['install_loss_factor']
        outputs[Dynamic.Mission.THRUST] = (
            ctx
            * tipspd**2
            * diam_prop**2
            * inputs['density_ratio']
            / (1.515e06)
            * 364.76
            * (1.0 - install_loss_factor)
        )

        # avoid divide by zero when shaft power is zero
        calc_idx = np.where(inputs['power_coefficient'] > 1e-6)  # index where CP > 1e-5
        prop_eff = np.zeros(self.options['num_nodes'])
        prop_eff[calc_idx] = (
            inputs['advance_ratio'][calc_idx]
            * ctx[calc_idx]
            / inputs['power_coefficient'][calc_idx]
        )
        outputs['propeller_efficiency'] = prop_eff
        outputs['install_efficiency'] = outputs['propeller_efficiency'] * (
            1.0 - install_loss_factor
        )

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        XFT = inputs['comp_tip_loss_factor']
        ctx = inputs['thrust_coefficient'] * XFT
        diam_prop = inputs[Aircraft.Engine.PROPELLER_DIAMETER]
        install_loss_factor = inputs['install_loss_factor']
        tipspd = inputs[Dynamic.Mission.PROPELLER_TIP_SPEED]

        unit_conversion_factor = 364.76 / 1.515e06
        partials["thrust_coefficient_comp_loss", 'thrust_coefficient'] = XFT
        partials["thrust_coefficient_comp_loss", 'comp_tip_loss_factor'] = inputs[
            'thrust_coefficient'
        ]
        partials[Dynamic.Mission.THRUST, 'thrust_coefficient'] = (
            XFT
            * tipspd**2
            * diam_prop**2
            * inputs['density_ratio']
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Mission.THRUST, 'comp_tip_loss_factor'] = (
            inputs['thrust_coefficient']
            * tipspd**2
            * diam_prop**2
            * inputs['density_ratio']
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Mission.THRUST, Dynamic.Mission.PROPELLER_TIP_SPEED] = (
            2
            * ctx
            * tipspd
            * diam_prop**2
            * inputs['density_ratio']
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Mission.THRUST, Aircraft.Engine.PROPELLER_DIAMETER] = (
            2
            * ctx
            * tipspd**2
            * diam_prop
            * inputs['density_ratio']
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Mission.THRUST, 'density_ratio'] = (
            ctx
            * tipspd**2
            * diam_prop**2
            * unit_conversion_factor
            * (1.0 - install_loss_factor)
        )
        partials[Dynamic.Mission.THRUST, 'install_loss_factor'] = (
            -ctx
            * tipspd**2
            * diam_prop**2
            * inputs['density_ratio']
            * unit_conversion_factor
        )

        calc_idx = np.where(inputs['power_coefficient'] > 1e-6)
        pow_coeff = inputs['power_coefficient']
        adv_ratio = inputs['advance_ratio']

        deriv_propeff_adv = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_ct = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_tip = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_propeff_cp = np.zeros(nn, dtype=pow_coeff.dtype)

        deriv_propeff_adv[calc_idx] = ctx[calc_idx] / pow_coeff[calc_idx]
        deriv_propeff_ct[calc_idx] = (
            adv_ratio[calc_idx] * XFT[calc_idx] / pow_coeff[calc_idx]
        )
        deriv_propeff_tip[calc_idx] = (
            adv_ratio[calc_idx]
            * inputs['thrust_coefficient'][calc_idx]
            / pow_coeff[calc_idx]
        )
        deriv_propeff_cp[calc_idx] = (
            -adv_ratio[calc_idx] * ctx[calc_idx] / pow_coeff[calc_idx] ** 2
        )

        partials["propeller_efficiency", "advance_ratio"] = deriv_propeff_adv
        partials["propeller_efficiency", "thrust_coefficient"] = deriv_propeff_ct
        partials["propeller_efficiency", "comp_tip_loss_factor"] = deriv_propeff_tip
        partials["propeller_efficiency", "power_coefficient"] = deriv_propeff_cp

        deriv_insteff_adv = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_ct = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_tip = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_cp = np.zeros(nn, dtype=pow_coeff.dtype)
        deriv_insteff_lf = np.zeros(nn, dtype=pow_coeff.dtype)

        deriv_insteff_adv[calc_idx] = (
            ctx[calc_idx] / pow_coeff[calc_idx] * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_ct[calc_idx] = (
            adv_ratio[calc_idx]
            * XFT[calc_idx]
            / pow_coeff[calc_idx]
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_tip[calc_idx] = (
            adv_ratio[calc_idx]
            * inputs['thrust_coefficient'][calc_idx]
            / pow_coeff[calc_idx]
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_cp[calc_idx] = (
            -adv_ratio[calc_idx]
            * ctx[calc_idx]
            / pow_coeff[calc_idx] ** 2
            * (1.0 - install_loss_factor[calc_idx])
        )
        deriv_insteff_lf[calc_idx] = (
            -adv_ratio[calc_idx] * ctx[calc_idx] / pow_coeff[calc_idx]
        )

        partials["install_efficiency", "advance_ratio"] = deriv_insteff_adv
        partials["install_efficiency", "thrust_coefficient"] = deriv_insteff_ct
        partials["install_efficiency", "comp_tip_loss_factor"] = deriv_insteff_tip
        partials["install_efficiency", "power_coefficient"] = deriv_insteff_cp
        partials["install_efficiency", 'install_loss_factor'] = deriv_insteff_lf
