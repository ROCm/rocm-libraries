# Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from itertools import product
from perflib.generators import Problem
from perflib.generators import RadixProblemGenerator

from sympy import sieve

import numpy as np

import perflib.specs

all_precisions = ['single', 'double']
all_directions = [-1, 1]
all_inplaces = [True, False]
all_reals = [True, False]
def_tuning_min_wgs = 64
def_tuning_max_wgs = 512
def_export_full_token = False
default_ngpus = 1
def_mp_size = 1
def_mp_exec = '/usr/bin/mpirun'
def_ingrid = [1, 1, 1]
def_outgrid = [1, 1, 1]

# yapf: disable
lengths = {

    'md': [
        (100,100,100),
        (160,160,168),
        (160,168,168),
        (160,168,192),
        (160,72,72),
        (160,80,72),
        (160,80,80),
        (168,168,192),
        (168,192,192),
        (168,80,80),
        (192,192,192),
        (192,192,200),
        (192,200,200),
        (192,84,84),
        (192,96,84),
        (192,96,96),
        (200,100,96),
        (200,200,200),
        (200,96,96),
        (208,100,100),
        (216,104,100),
        (216,104,104),
        (224,104,104),
        (224,108,104),
        (224,108,108),
        (240,108,108),
        (240,112,108),
        (240,112,112),
        (280,128,128),
        (60,60,60),
        (64,64,52),
        (64,64,64),
        (72,72,52),
        (72,72,72),
        (80,80,80),
        (84,84,72),
        (96,96,96),

        (108,108,80),
        (216,216,216),

        (128,128,256),
        (240,224,224),
        (80,84,14),
        (80,84,144),

        (25,20,20),
        (42,32,32),
        (75,55,55),
    ],

    'misc3d': [
        (256, 256, 256),
        (336, 336, 56),
    ],

    'misc2d': [
        (256, 256),
        (56, 336),
        (4096, 4096),
        (336, 18816),
    ],

    'simpleL1D': [
        6561,
        8192,
        10000,
        16384,
        32768,
        40000,
        65536,
    ],

    'large1d': [
        8192,
        10000,
        10752,
        15625,
        16384,
        16807,
        18816,
        19683,
        21504,
        32256,
        43008,
    ],

    'large1DExtended': [
        4096, 4704, 4913, 5488, 6144, 6561, 8192,
        9216, 10000, 10240, 10752, 11200, 12288,
        15625, 16384, 16807, 17576, 18816, 19200,
        19683, 20480, 21504, 21952, 23232, 24576,
        26000, 28672, 32256, 32768, 34969, 36864,
        38880, 40000, 40960, 43008, 46080, 48000,
        49152, 51200, 53248, 57344, 65536, 68600,
        71344, 73984, 76832, 79860, 81920, 83521,
        87808, 95832, 98304, 102400, 106496,
        110592, 114688, 131072, 262144,
    ],

    'small1d': [
        24,
        48,
        52,
        60,
        64,
        68,
        72,
        80,
        96,
        100,
        128,
        168,
        200,
        256,
        280,
        330,
        336,
        512,
    ],

    'mixed': [
        225, 240, 300, 486, 600, 900, 958, 1014, 1139,
        1250, 1427, 1463, 1480, 1500, 1568, 1608, 1616, 1638, 1656,
        1689, 1696, 1708, 1727, 1744, 1752, 1755, 1787, 1789, 1828,
        1833, 1845, 1860, 1865, 1875, 1892, 1897, 1899, 1900, 1903,
        1905, 1912, 1933, 1938, 1951, 1952, 1954, 1956, 1961, 1964,
        1976, 1997, 2004, 2005, 2006, 2012, 2016, 2028, 2033, 2034,
        2038, 2069, 2100, 2113, 2116, 2123, 2136, 2152, 2160, 2167,
        2181, 2182, 2187, 2205, 2208, 2242, 2250, 2251, 2288, 2306,
        2342, 2347, 2352, 2355, 2359, 2365, 2367, 2383, 2385, 2387,
        2389, 2429, 2439, 2445, 2448, 2462, 2467, 2474, 2478, 2484,
        2486, 2496, 2500, 2503, 2519, 2525, 2526, 2533, 2537, 2556,
        2558, 2559, 2566, 2574, 2576, 2594, 2604, 2607, 2608, 2612,
        2613, 2618, 2632, 2635, 2636, 2641, 2652, 2654, 2657, 2661,
        2663, 2678, 2688, 2690, 2723, 2724, 2728, 2729, 2733, 2745,
        2755, 2760, 2772, 2773, 2780, 2786, 2789, 2790, 2805, 2807,
        2808, 2812, 2815, 2816, 2820, 2826, 2830, 2834, 2841, 2847,
        2848, 2850, 2852, 2853, 2872, 2877, 2882, 2883, 2886, 2887,
        2892, 2893, 2917, 2922, 2924, 2926, 2928, 2929, 2932, 2933,
        2934, 2938, 2951, 2960, 2970, 2979, 2990, 2994, 2998, 2999,
        3000, 3001, 3003, 3004, 3008, 3034, 3035, 3039, 3040, 3042,
        3048, 3052, 3055, 3060, 3065, 4000, 12000, 24000,
    ],

    'generated': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
                   20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 36, 40, 42, 44, 45,
                   48, 49, 50, 52, 54, 56, 60, 64, 72, 75, 80, 81, 84, 88, 90,
                   96, 100, 104, 108, 112, 120, 121, 125, 128, 135, 144, 150,
                   160, 162, 168, 169, 176, 180, 192, 200, 208, 216, 224, 225,
                   240, 243, 250, 256, 270, 288, 300, 320, 324, 336, 343, 360,
                   375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576,
                   600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864,
                   900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215,
                   1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620,
                   1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187,
                   2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
                   3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750,
                   3840, 3888, 4000, 4050, 4096],

    'qa1d10b': [
        16777216,
        14348907,
        9765625,
    ],

    'qa2d10b': [
        (3125, 3125),
        (4096, 4096),
        (6561, 6561),
    ],

    'qa3d10b': [
        (256, 256, 256),
        (243, 243, 243),
        (125, 125, 125),
    ],

    'qaReal3d10b': [
        (100, 100, 100),
        (200, 200, 200),
        (192, 192, 192),
    ],

    'nonSupported1D': [
        38, 46, 57, 58, 62, 69, 74, 76, 82, 86, 87, 92, 93, 94, 95, 106, 111,
        114, 115, 116, 118, 122, 123, 124, 129, 133, 134, 138, 141, 142, 145,
        146, 148, 152, 155, 158, 159, 161, 164, 166, 171, 172, 174, 177, 178,
        183, 184, 185, 186, 188, 190, 194, 201, 202, 203, 205, 206, 207, 209,
        212, 213, 214, 215, 217, 218, 219, 222, 226, 228, 230, 232, 235, 236,
        237, 244, 246, 247, 248, 249, 253, 254, 258, 259, 261, 262, 265, 266,
        267, 268, 274, 276, 278, 279, 282, 284, 285, 287, 290, 291, 292, 295,
        296, 298, 299, 301, 302, 303, 304, 305, 309, 310, 314, 316, 318, 319,
        321, 322, 323, 326, 327, 328, 329, 332, 333, 334, 335, 339, 341, 342,
        344, 345, 346, 348, 354, 355, 356, 358, 361, 362, 365, 366, 368, 369,
        370, 371, 372, 376, 377, 380, 381, 382, 386, 387, 388, 391, 393, 394,
        395, 398, 399, 402, 403, 404, 406, 407, 410, 411, 412, 413, 414, 415,
        417, 418, 422, 423, 424, 426, 427, 428, 430, 434, 435, 436, 437, 438,
        444, 445, 446, 447, 451, 452, 453, 454, 456, 458, 460, 464, 465, 466,
        469, 470, 471, 472, 473, 474, 475, 477, 478, 481, 482, 483, 485, 488,
        489, 492, 493, 494, 496, 497, 498, 501, 502, 505, 506, 508, 511, 513,
        514, 515, 516, 517, 518, 519, 522, 524, 526, 527, 529, 530, 531, 532,
        533, 534, 535, 536, 537, 538, 542, 543, 545, 548, 549, 551, 552, 553,
        554, 555, 556, 558, 559, 562, 564, 565, 566, 568, 570, 573, 574, 575,
        579, 580, 581, 582, 583, 584, 586, 589, 590, 591, 592, 596, 597, 598,
        602, 603, 604, 606, 608, 609, 610, 611, 614, 615, 618, 620, 621, 622,
        623, 626, 627, 628, 629, 632, 633, 634, 635, 636, 638, 639, 642, 644,
        645, 646, 649, 651, 652, 654, 655, 656, 657, 658, 662, 664, 665, 666,
        667, 668, 669, 670, 671, 674, 678, 679, 681, 682, 684, 685, 687, 688,
        689, 690, 692, 694, 695, 696, 697, 698, 699, 703, 705, 706, 707, 708,
        710, 711, 712, 713, 716, 717, 718, 721, 722, 723, 724, 725, 730, 731,
        732, 734, 736, 737, 738, 740, 741, 742, 744, 745, 746, 747, 749, 752,
        753, 754, 755, 758, 759, 760, 762, 763, 764, 766, 767, 771, 772, 774,
        775, 776, 777, 778, 779, 781, 782, 783, 785, 786, 788, 789, 790, 791,
        793, 794, 795, 796, 798, 799, 801, 802, 803, 804, 805, 806, 807, 808,
        812, 813, 814, 815, 817, 818, 820, 822, 824, 826, 828, 830, 831, 834,
        835, 836, 837, 838, 841, 842, 843, 844, 846, 848, 849, 851, 852, 854,
        855, 856, 860, 861, 862, 865, 866, 868, 869, 870, 871, 872, 873, 874,
        876, 878, 879, 885, 886, 888, 889, 890, 892, 893, 894, 895, 897, 898,
        899, 901, 902, 903, 904, 905, 906, 908, 909, 912, 913, 914, 915, 916,
        917, 920, 921, 922, 923, 925, 926, 927, 928, 930, 931, 932, 933, 934,
        938, 939, 940, 942, 943, 944, 946, 948, 949, 950, 951, 954, 955, 956,
        957, 958, 959, 961, 962, 963, 964, 965, 966, 969, 970, 973, 974, 976,
        978, 979, 981, 982, 984, 985, 986, 987, 988, 989, 992, 993, 994, 995,
        996, 998, 999, 1002, 1003, 1004, 1005, 1006, 1007, 1010, 1011, 1012,
        1015, 1016, 1017, 1018, 1022, 1023, 1025, 1026, 1027, 1028, 1030,
        1032, 1034, 1035, 1036, 1037, 1038, 1041, 1042, 1043, 1044, 1045,
        1046, 1047, 1048, 1052, 1054, 1055, 1057, 1058, 1059, 1060, 1062,
        1064, 1065, 1066, 1067, 1068, 1070, 1072, 1073, 1074, 1075, 1076,
        1077, 1079, 1081, 1082, 1083, 1084, 1085, 1086, 1090, 1094, 1095, 1096,
        1098, 1099, 1101, 1102, 1104, 1106, 1107, 1108, 1110, 1111, 1112, 1113,
        1114, 1115, 1116, 1118, 1119, 1121, 1124, 1126, 1127, 1128, 1130, 1131,
        1132, 1133, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1145,
        1146, 1147, 1148, 1149, 1150, 1154, 1157, 1158, 1159, 1160, 1161, 1162,
        1164, 1165, 1166, 1167, 1168, 1169, 1172, 1173, 1174, 1175, 1177, 1178,
        1179, 1180, 1182, 1184, 1185, 1186, 1189, 1191, 1192, 1194, 1195, 1196,
        1197, 1198, 1199, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1211,
        1212, 1214, 1216, 1218, 1219, 1220, 1221, 1222, 1226, 1227, 1228, 1230,
        1233, 1234, 1235, 1236, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245,
        1246, 1247, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1261, 1262,
        1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1276,
        1278, 1281, 1282, 1284, 1285, 1286, 1288, 1290, 1292, 1293, 1294, 1295,
        1298, 1299, 1302, 1304, 1305, 1306, 1308, 1310, 1311, 1312, 1313, 1314,
        1315, 1316, 1317, 1318, 1322, 1324, 1325, 1328, 1329, 1330, 1332, 1333,
        1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1345, 1346,
        1347, 1348, 1349, 1351, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1362,
        1363, 1364, 1366, 1368, 1369, 1370, 1371, 1374, 1376, 1378, 1379, 1380,
        1382, 1383, 1384, 1385, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394,
        1395, 1396, 1397, 1398, 1401, 1402, 1403, 1405, 1406, 1407, 1410, 1411,
        1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1424,
        1425, 1426, 1431, 1432, 1434, 1435, 1436, 1437, 1438, 1441, 1442, 1443,
        1444, 1446, 1448, 1449, 1450, 1454, 1455, 1457, 1460, 1461, 1462, 1463,
        1464, 1465, 1466, 1467, 1468, 1469, 1472, 1473, 1474, 1475, 1476, 1477,
        1478, 1479, 1480, 1482, 1484, 1486, 1488, 1490, 1491, 1492, 1494, 1495,
        1497, 1498, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510,
        1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1522, 1524, 1525, 1526,
        1527, 1528, 1529, 1532, 1533, 1534, 1535, 1537, 1538, 1539, 1541, 1542,
        1544, 1545, 1546, 1548, 1550, 1551, 1552, 1554, 1555, 1556, 1557, 1558,
        1561, 1562, 1563, 1564, 1565, 1566, 1569, 1570, 1572, 1574, 1576, 1577,
        1578, 1580, 1581, 1582, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592,
        1593, 1594, 1595, 1596, 1598, 1599, 1602, 1603, 1604, 1605, 1606, 1608,
        1610, 1611, 1612, 1614, 1615, 1616, 1618, 1622, 1623, 1624, 1626, 1628,
        1629, 1630, 1631, 1633, 1634, 1635, 1636, 1639, 1640, 1641, 1642, 1643,
        1644, 1645, 1646, 1647, 1648, 1649, 1651, 1652, 1653, 1654, 1655, 1656,
        1658, 1659, 1660, 1661, 1662, 1665, 1668, 1670, 1671, 1672, 1673, 1674,
        1675, 1676, 1677, 1678, 1679, 1681, 1682, 1684, 1685, 1686, 1687, 1688,
        1689, 1691, 1692, 1695, 1696, 1698, 1702, 1703, 1704, 1705, 1706, 1707,
        1708, 1710, 1711, 1712, 1713, 1714, 1717, 1718, 1719, 1720, 1722, 1724,
        1725, 1726, 1727, 1729, 1730, 1731, 1732, 1735, 1736, 1737, 1738, 1739,
        1740, 1742, 1743, 1744, 1745, 1746, 1748, 1749, 1751, 1752, 1754, 1756,
        1757, 1758, 1761, 1762, 1763, 1765, 1766, 1767, 1769, 1770, 1771, 1772,
        1773, 1774, 1775, 1776, 1778, 1779, 1780, 1781, 1784, 1786, 1788, 1790,
        1791, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1802, 1803, 1804, 1805,
        1806, 1807, 1808, 1809, 1810, 1812, 1813, 1814, 1816, 1817, 1818, 1819,
        1821, 1822, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1832, 1833, 1834,
        1835, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1849,
        1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1860, 1862, 1863,
        1864, 1865, 1866, 1868, 1869, 1874, 1876, 1878, 1880, 1881, 1882, 1883,
        1884, 1885, 1886, 1887, 1888, 1891, 1892, 1893, 1894, 1895, 1896, 1897,
        1898, 1899, 1900, 1902, 1903, 1905, 1906, 1908, 1909, 1910, 1912, 1914,
        1915, 1916, 1917, 1918, 1919, 1921, 1922, 1923, 1924, 1926, 1927, 1928,
        1929, 1930, 1932, 1934, 1935, 1937, 1938, 1939, 1940, 1941, 1942, 1943,
        1945, 1946, 1947, 1948, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
        1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972,
        1974, 1975, 1976, 1977, 1978, 1981, 1982, 1983, 1984, 1985, 1986, 1988,
        1990, 1991, 1992, 1994, 1995, 1996, 1998, 2001, 2004, 2005, 2006, 2007,
        2008, 2009, 2010, 2012, 2013, 2014, 2015, 2018, 2019, 2020, 2021, 2022,
        2024, 2026, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2041,
        2042, 2043, 2044, 2045, 2046, 2047,
     ],

    'mgpu': [
        (256),
        (256, 256),
        (256, 256, 256),
        (128, 256, 512),
        (512, 512, 512),
        (630, 630, 630),
    ],

    'mpi': [
        (128, 256, 512),
        (256, 256, 256),
        (378, 189, 63),
        (512, 512, 512),
        (630, 310, 630),
    ],

    'scaling2D': [
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (8192, 8192),
    ],

    'scaling3D': [
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],

}
# yapf: enable


def mktag(tag, dimension, precision, direction, inplace, real):
    t = [
        tag,
        str(dimension) + 'D', precision, {
            -1: 'forward',
            1: 'backward'
        }[direction], {
            True: 'real',
            False: 'complex'
        }[real], {
            True: 'in-place',
            False: 'out-of-place'
        }[inplace]
    ]
    return "_".join(t)


# yield problem sizes with default precision, direction, etc
def default_length_params(tag, lengths, nbatch, ngpus=default_ngpus, mp_size=def_mp_size, \
                          mp_exec=def_mp_exec, precisions=all_precisions, \
                          directions=all_directions, inplaces=all_inplaces, \
                          reals=all_reals, min_wgs=def_tuning_min_wgs, \
                          max_wgs=def_tuning_max_wgs,  ingrid=def_ingrid, outgrid=def_outgrid, \
                          full_token=def_export_full_token, meta = {}):

    for precision, direction, inplace, real in product(precisions, directions,
                                                       inplaces, reals):
        for length in lengths:
            length = (length, ) if isinstance(length, int) else length

            yield Problem(length,
                          tag=mktag(tag, len(length), precision, direction,
                                    inplace, real),
                          nbatch=nbatch,
                          ngpus=ngpus,
                          mp_size=mp_size,
                          mp_exec=mp_exec,
                          ingrid=ingrid,
                          outgrid=outgrid,
                          direction=direction,
                          inplace=inplace,
                          real=real,
                          precision=precision,
                          min_wgs=min_wgs,
                          max_wgs=max_wgs,
                          full_token=full_token,
                          meta=meta)


def md():
    """Molecular dynamics suite."""

    yield from default_length_params("md", lengths['md'], 10)


def qa():
    """AMD QA suite."""

    for length1 in [
            8192, 10752, 15625, 16384, 16807, 18816, 19683, 21504, 32256, 43008
    ]:
        for direction in [-1, 1]:
            yield Problem([length1],
                          tag=mktag("qa1", 1, 'double', direction, False,
                                    False),
                          nbatch=10000,
                          direction=direction,
                          inplace=False,
                          real=False,
                          precision='double')

    yield Problem([10000],
                  tag=mktag('qa10k', 1, 'double', 1, False, False),
                  nbatch=10000,
                  direction=1,
                  inplace=False,
                  real=False,
                  precision='double')

    yield from default_length_params("336x336x56_b1", [(336, 336, 56)],
                                     1,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[False])

    yield from default_length_params("336x336x56_b10", [(336, 336, 56)],
                                     10,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[False])

    for length3 in lengths['md']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag('qa3md', 3, 'single', direction, False,
                                    True),
                          nbatch=10,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single',
                          meta={'figtype': 'bargraph'})

    for length in lengths['qa1d10b']:
        yield Problem([length],
                      tag=mktag("qa1d10b", 1, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length2 in lengths['qa2d10b']:
        yield Problem(length2,
                      tag=mktag("qa2d10b", 2, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length3 in lengths['qa3d10b']:
        yield Problem(length3,
                      tag=mktag("qa3d10b", 3, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length3 in lengths['qaReal3d10b']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag("qaReal3d10b", 3, 'single', direction,
                                    False, True),
                          nbatch=10,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single')


def misc2d():
    """Miscellaneous 2D sizes."""

    yield from default_length_params("misc2d", lengths['misc2d'], 1)


def misc3d():
    """Miscellaneous 3D sizes."""

    yield from default_length_params("misc3d", lengths['misc3d'], 1)


def simpleL1D():
    """Basic C2C Large 1D sizes."""

    yield from default_length_params("C2C_L1D",
                                     lengths['simpleL1D'],
                                     8000,
                                     reals=[False])


def large1d():
    """Large 1D sizes."""

    yield from default_length_params("large1d",
                                     lengths['large1d'],
                                     10000,
                                     reals=[False])


def generated1d(skip=1):
    """Explicitly generated 1D lengths."""

    yield from default_length_params("generated1d",
                                     lengths['generated'][::skip], 1000)


def generated2d():
    """Explicitly generated 2D lengths."""

    lengths2d = list(filter(lambda x: x <= 1024, lengths['generated']))
    yield from default_length_params("generated2d", lengths2d, 100)


def generated3d():
    """Explicitly generated 3D lengths."""

    lengths3d = list(filter(lambda x: x <= 512, lengths['generated']))
    yield from default_length_params("generated3d", lengths3d, 1)


def prime():
    """Large selection of prime lengths."""

    yield from default_length_params("prime", list(sieve.primerange(11, 1000)),
                                     10000)


def mixed1d(skip=1):
    """Mixed 1D lengths."""

    yield from default_length_params("mixed", lengths['mixed'][::skip], 10000)


def prime_limited():
    """Limited selection of prime lengths for regular testing."""

    yield from default_length_params("prime", [23, 521, 997], 10000)


def small_prime_extended():
    """Extended selection of small prime lengths."""

    yield from default_length_params("small_prime_extended",
                                     list(sieve.primerange(32, 8192)), 10000)


def large_prime_extended():
    """Extended selection of large prime lengths."""

    yield from default_length_params("large_prime_extended",
                                     list(sieve.primerange(8192, 65536)), 1000)


def prime_2D():
    """Selection of 2D prime lengths."""

    primes = np.array(list(sieve.primerange(32, 256)))
    combined2d = np.vstack((primes, primes)).T

    yield from default_length_params("prime_2D", combined2d, 1000)


def prime_3D():
    """Selection of 3D prime lengths."""

    primes = np.array(list(sieve.primerange(32, 256)))
    combined3d = np.vstack((primes, primes, primes)).T

    yield from default_length_params("prime_3D", combined3d, 100)


def non_supported_lengths_1D():
    """Non-prime 1D lengths in (32,2048) that default to Bluestein.
       Subject to change as further kernel support is added."""

    yield from default_length_params("non_supported_lengths_1D",
                                     lengths['nonSupported1D'], 10000)


def mpi():
    """MPI sizes."""

    # Grid sizes can be overwritten from CLI
    yield from default_length_params("mpi",
                                     lengths['mpi'],
                                     1,
                                     precisions=['single'],
                                     reals=[False])


def mgpu():
    """Multi-GPU sizes."""

    yield from default_length_params("mgpu",
                                     lengths['mgpu'],
                                     1,
                                     precisions=['single'],
                                     reals=[False])


def strongScaling():
    """Strong scalability test sizes."""

    for length in lengths['scaling2D']:
        direction = -1
        precision = 'double'
        nbatch = 1
        yield Problem(length,
                      tag=mktag(
                          'strongScaling' + "x".join([str(x) for x in length]),
                          nbatch, precision, direction, False, False),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=False,
                      real=False,
                      precision='double',
                      meta={'scaling': 'strong'})

    for length in lengths['scaling3D']:
        direction = -1
        precision = 'double'
        nbatch = 1
        yield Problem(length,
                      tag=mktag(
                          'strongScaling' + "x".join([str(x) for x in length]),
                          nbatch, precision, direction, False, False),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=False,
                      real=False,
                      precision='double',
                      meta={'scaling': 'strong'})


def weakScaling():
    """Weak scalability test sizes."""

    for length in lengths['scaling2D']:
        direction = -1
        precision = 'double'
        nbatch = 1
        yield Problem(length,
                      tag=mktag(
                          'weakScaling' + "x".join([str(x) for x in length]),
                          nbatch, precision, direction, False, False),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=False,
                      real=False,
                      precision='double',
                      meta={'scaling': 'weak'})

    for length in lengths['scaling3D']:
        direction = -1
        precision = 'double'
        nbatch = 1
        yield Problem(length,
                      tag=mktag(
                          'weakScaling' + "x".join([str(x) for x in length]),
                          nbatch, precision, direction, False, False),
                      nbatch=nbatch,
                      direction=direction,
                      inplace=False,
                      real=False,
                      precision='double',
                      meta={'scaling': 'weak'})


def unbatched_1d():
    """All tested single-batch 1D transforms."""

    yield from simpleL1D()
    yield from large1d()
    yield from generated1d(2)
    yield from mixed1d(2)
    yield from prime_limited()


def batched_1d():
    for subtag in ['simpleL1D', 'large1d']:  #, 'generated', 'mixed']:
        for precision, direction, inplace, real in product(
                all_precisions, all_directions, all_inplaces, all_reals):
            subcaption = precision
            subcaption += " forward " if direction == -1 else " backward "
            subcaption += " real" if real else " complex"
            subcaption += " in-place " if inplace else " out-of-place "
            subcaption += " length $N$ batch size $N$."

            for length in lengths[subtag]:
                yield Problem([length],
                              tag=mktag('batched_1d_contiguous_' + subtag, 1,
                                        precision, direction, inplace, real),
                              nbatch=length,
                              inplace=inplace,
                              precision=precision,
                              real=real,
                              meta={'caption': 'Batched 1D ' + subcaption})
                ncomplex = length // 2 + 1
                istride = length
                ostride = length
                if real:
                    if direction == -1:
                        istride = 2 * ncomplex if inplace else length
                        ostride = ncomplex
                    else:
                        istride = ncomplex
                        ostride = 2 * ncomplex if inplace else length

                yield Problem(
                    [length],
                    tag=mktag('batched_1d_strided_' + subtag, 1, precision,
                              direction, inplace, real),
                    nbatch=length,
                    istride=istride,
                    ostride=ostride,
                    idist=1,
                    odist=1,
                    inplace=inplace,
                    precision=precision,
                    real=real,
                    meta={'caption': 'Batched strided 1D ' + subcaption})


def batched_1d_small_r2c():
    """Small 1D sizes, large batch size."""

    yield from default_length_params("small1d",
                                     lengths['small1d'],
                                     10000,
                                     reals=[True])


def batch_const_count():
    # batch * length = 2^25 ... 2^30
    for direction in [-1, 1]:
        for precision in all_precisions:
            for exp in [25, 26, 27, 28, 29, 30, 31, 32]:
                for place in all_inplaces:
                    for lexp in range(4, exp + 1):
                        length = 2**lexp
                        batch = 2**(exp - lexp)

                        yield Problem([length],
                                      tag=mktag("footprint2exp" + str(exp), 1,
                                                precision, direction, False,
                                                False),
                                      nbatch=batch,
                                      direction=direction,
                                      inplace=place,
                                      real=False,
                                      precision=precision)


def benchmarks():
    """Benchmarks: XXX"""

    pow2 = [2**k for k in range(30)]
    pow3 = [3**k for k in range(19)]
    minmax = {
        1: (256, 536870912),
        2: (64, 32768),
        3: (16, 1024),
    }

    all_lengths = sorted(pow2 + pow3)
    dimensions = [1, 2, 3]

    for dimension in dimensions:
        min1, max1 = minmax[dimension]
        lengths = [(3 * [length])[:dimension] for length in all_lengths
                   if min1 <= length <= max1]
        yield from default_length_params('benchmark', lengths, 1)


def all():
    """All suites run during regular testing."""

    yield from benchmarks()

    # pow 5, 7 (benchmarks does pow 2, 3)
    yield from default_length_params("pow5", [5**k for k in range(4, 8)], 5000)
    yield from default_length_params("pow7", [7**k for k in range(3, 7)], 5000)

    yield from md()

    yield from prime_limited()

    yield from large1d()
    yield from misc2d()
    yield from misc3d()


def short_test():
    """A few small sizes for script testing."""

    yield from default_length_params("short_test", [(8), (16), (32),
                                                    (4294967296)],
                                     1,
                                     reals=[False])


def tuning_example():
    """tuning 3 examples problems"""

    # real 1d : odd/even/small/large - fwd
    length1s = [243, 486, 16807, 16384]
    yield from default_length_params("real_1d_fwd",
                                     length1s,
                                     40000,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[True],
                                     min_wgs=128,
                                     max_wgs=256)

    # real 1d : odd/even/small/large - bwd - FIXME: bugs: batchsize for bwd can't exceed 1024
    yield from default_length_params("real_1d_fwd",
                                     length1s,
                                     1024,
                                     directions=[1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[True],
                                     min_wgs=128,
                                     max_wgs=256)

    # real 2d
    length2s = [(55, 55), (64, 52), (52, 52)]
    yield from default_length_params("real_2d",
                                     length2s,
                                     10000,
                                     directions=[-1, 1],
                                     precisions=['double'],
                                     inplaces=[False],
                                     reals=[True],
                                     min_wgs=128,
                                     max_wgs=256)

    # real 3d
    length3s = [(75, 55, 55), (208, 104, 104), (100, 100, 100),
                (200, 200, 200)]
    yield from default_length_params("real_3d",
                                     length3s,
                                     10,
                                     directions=[-1, 1],
                                     precisions=['double'],
                                     inplaces=[False],
                                     reals=[True],
                                     min_wgs=128,
                                     max_wgs=256)

    # complex
    yield from default_length_params("81_1d", [(81)],
                                     60000,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[False],
                                     reals=[False],
                                     min_wgs=128,
                                     max_wgs=256)

    yield from default_length_params("81_2d", [(81, 81)],
                                     8000,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[False],
                                     reals=[False],
                                     min_wgs=128,
                                     max_wgs=256)

    # batch=500 to enabling tuning with intrinsic buffer
    yield from default_length_params("81_3d", [(81, 81, 81)],
                                     500,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[False],
                                     reals=[False],
                                     min_wgs=128,
                                     max_wgs=256)


def tuning_2D_example():
    """tuning 3 examples problems"""

    lengths = [(81, 32), (81, 25), (125, 32)]

    yield from default_length_params("2D_SINGLE",
                                     lengths,
                                     10000,
                                     directions=[-1],
                                     precisions=['single'],
                                     inplaces=[False],
                                     reals=[False],
                                     max_wgs=1024)


def tuning_suite():
    """tuning"""

    # basically, when tuning for single, we can tune wgs range from 128~512, for double, wgs range is 128~256
    # but you can also change it for particular problem.
    # But inside our cpp tuner implementation, min_wgs might still be automatically changed
    # if the setting gives no any candidate (example, a len64 with min_wgs=128 might not derive any)

    # complex transforms in suite qa.
    for length1 in [
            8192, 10000, 10752, 15625, 16384, 16807, 18816, 19683, 21504
    ]:
        for direction in [1]:
            yield Problem([length1],
                          tag=mktag("qa1", 1, 'double', direction, False,
                                    False),
                          nbatch=10000,
                          direction=direction,
                          inplace=False,
                          real=False,
                          precision='double',
                          min_wgs=128,
                          max_wgs=256)

    # batch=5000 to enabling tuning with intrinsic buffer
    # since batch 10000 causes memory offset > 2^32, buffer inst will be disabled
    for length1 in [32256, 43008]:
        for direction in [1]:
            yield Problem([length1],
                          tag=mktag("qa1", 1, 'double', direction, False,
                                    False),
                          nbatch=5000,
                          direction=direction,
                          inplace=False,
                          real=False,
                          precision='double',
                          min_wgs=128,
                          max_wgs=256)

    # we'd like to search more for this problem, so min_wgs = 64, not 128
    yield from default_length_params("336x336x56_b1", [(336, 336, 56)],
                                     1,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[False],
                                     max_wgs=256,
                                     full_token=True)

    yield from default_length_params("336x336x56_b10", [(336, 336, 56)],
                                     10,
                                     directions=[-1],
                                     precisions=['double'],
                                     inplaces=[True, False],
                                     reals=[False],
                                     max_wgs=256)

    for length3 in lengths['md']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag('qa3md', 3, 'single', direction, False,
                                    True),
                          nbatch=10,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single',
                          min_wgs=128,
                          max_wgs=256)

    for length in lengths['qa1d10b']:
        yield Problem([length],
                      tag=mktag("qa1d10b", 1, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single',
                      min_wgs=128)

    for length2 in lengths['qa2d10b']:
        yield Problem(length2,
                      tag=mktag("qa2d10b", 2, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single',
                      min_wgs=128)

    for length3 in lengths['qa3d10b']:
        yield Problem(length3,
                      tag=mktag("qa3d10b", 3, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single',
                      min_wgs=128)


def partial_pass():
    for length in [(64, 64, 128), (64, 64, 64), (64, 64, 52), (60, 60, 60),
                   (32, 32, 128), (32, 32, 64), (64, 32, 128)]:
        for direction in [-1, 1]:
            for precision in ['single', 'double']:
                for place in all_inplaces:
                    for batch in [
                            1, 5, 20, 50, 100, 200, 500, 1000, 1500, 3000,
                            5000, 7500, 10000
                    ]:

                        yield Problem(length,
                                      tag=mktag("partial_pass", 1, precision,
                                                direction, place, False),
                                      nbatch=batch,
                                      direction=direction,
                                      inplace=place,
                                      real=False,
                                      meta={'ivariable': 'batch'},
                                      precision=precision)


def large_1d_extended():
    """All supported L1D_CC compute scheme sizes"""

    for length in lengths['large1DExtended']:
        for direction in [-1, 1]:
            for precision in ['single', 'double']:
                for place in all_inplaces:
                    for batch in [1, 10, 100, 1000, 10000]:
                        yield Problem([length],
                                      tag=mktag("large_1d_extended", 1,
                                                precision, direction, place,
                                                False),
                                      nbatch=batch,
                                      direction=direction,
                                      inplace=place,
                                      real=False,
                                      meta={'ivariable': 'batch'},
                                      precision=precision)
