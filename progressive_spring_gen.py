""" This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Written by: Mariusz Warzecha

    Date:14.02.2023

    Description: This script allows for the generation of a progressive spring geometry.
                The generation process is divided into two steps. First, basing on user
                input, the script generates coordinates of points defining the spring
                center curve. In the second step those points are used to generate 3D
                parametric software code (using defined API)  allowing creation of the 
                spring geometry in external program.
                Currently two programs are supported: Ansys SpaceClaim and Gmsh.
                For more details on used algorithm, please refer to article: ...........

    Usage: To generate progressive spring geometry, the user should give the parameters
         required and described in "USER INTERFACE" section of the script. Then the script
         should be run with Python 3 interpreter and the resulting file should be used
         to obtain the spring geometry in SpaceClaim or Gmsh (depending on user choise)
"""

import numpy as np
import math
import matplotlib.pylab as plt  # for test purpose

# ***************************************************************************************
# USER INTERFACE
# ***************************************************************************************

# script saves output in a file stored in the folder given below
file_store_dir = '/home/mariusz/Sym_Sprezyna_P_Baran/Geometria_sprezyn/'

# this variable defines software, in which spring geometry will be generated, currently
# it can take two values, either 'Gmsh' or 'SpaceClaim'
geom_output = 'Gmsh'

# script allows for two spring generation approaches: the user specifies radiuses between
# coils with different pitch - the algo_type variable should then take value 'ControlRadius'
# or the user defines the spting height - the algo_type variable should take value 'ControlHeight'
algo_type = 'ControlHeight'

# dictionary param_control_radius contains parameters required for spring geometry generation
# in case when algo_type = 'ControlRadius'
# parameters description:
# D - spring diameter [mm]
# d - spring wire diameter [mm]
# delta - list containing slopes for colis [-]
# coils - list containing coil number for each slope, its length should be equal to the length
#         of list containing coil slopes [-]
# radiuses - list defining radiuses between slope changes, its length should be equal to the
#         lenght of list containing coil slopes minus one [mm]
# point_density - float number specifing distance between points describing spring central
#         curve (smaller value results in better curve representation but may be problematic
#         during geometry generation in 3D parametric software
param_control_radius = {'D': 50,
                        'd': 8,
                        'delta': [0.050885616, 0.2493, 0.050885616],
                        'coils': [1, 3.5, 1],
                        'radiuses': [500, 50],
                        'point_density': 0.1}

# dictionary param_control_height contains parameters required for spring geometry generation
# in case when algo_type = 'ControlHeight'
# parameters description:
# D - spring diameter [mm]
# d - spring wire diameter [mm]
# radiuses - list defining radiuses between slope changes, its length should be equal to the
#             lenght of list containing coil slopes minus one [mm]
# coils - list containing coil number for each slope, its length should be equal to the length
#         of list containing coil slopes [-]
# spring_height - float number defining the spring height, based on this value slope for active
#                 coils will be calculated, to small value (smaller then number of coils *
#                 spring wire diameter) will make the spring geometry unfeasable [mm]
# point_density - float number specifing distance between points describing spring central
#         curve (smaller value results in better curve representation but may be problematic
#         during geometry generation in 3D parametric software
param_control_height = {'D': 50,
                        'd': 10,
                        'radiuses': [500, 50],
                        'coils': [1, 3.5, 1],
                        'spring_height': 160,
                        'point_density': 0.1}


# *********************************************************************************************
# PRIVATE IMPLEMENTATION - DO NOT CHANGE
# *********************************************************************************************

def gen_central_points_radius_control(parameters):
    """ Calculate coordinates of spring central curve for user defined radiuses and slopes.

    Parameters
    ----------
    parameters : dictionary
        The input dictionary should contain following variables: D - spring diameter [mm],
        d - spring wire diameter [mm], delta - list containing slopes for coils [-],
        coils - list containing coil number for each slope [-], radiuses - list containing
        radiuses between each slope change

    Returns
    -------
    2D numpy.array
        The array has 3 colums for x, y and z point coordinates. Each row is a separate point.

    Notes
    -----
    To be added if neccessary.

    References
    ----------
    Add reference to article when published.
    """
    D = parameters['D']
    R = D/2
    delta = parameters['delta']
    coils = parameters['coils']
    radiuses = parameters['radiuses']
    point_density = parameters['point_density']
    fi_A1 = 2 * math.pi * coils[0]
    beta = abs(delta[1] - delta[0])
    z_A1 = R * fi_A1 * math.tan(delta[0])
    x_A1 = R * fi_A1
    L_AS = radiuses[0] * math.tan(beta/2)
    delta_fi_AS = L_AS * math.cos(delta[0])
    delta_z_AS = L_AS * math.sin(delta[0])
    fi_S1 = fi_A1 + delta_fi_AS
    z_S1 = z_A1 + delta_z_AS
    L_cur = radiuses[0] * (delta[1] - delta[2])
    x_C1 = R * fi_A1 - radiuses[0] * math.sin(delta[0])
    y_C1 = R * fi_A1 * math.tan(delta[0]) + radiuses[0] * math.cos(delta[0])
    x_A1 = fi_A1 * R
    y_A1 = R * fi_A1 * math.tan(delta[0])
    x_S1 = R * fi_A1 + L_AS * math.cos(delta[0])
    y_S1 = R * fi_A1 * math.tan(delta[0]) + L_AS * math.sin(delta[0])
    x_A2 = x_A1 + 2 * math.pi * coils[1] * R
    delta_x_A2S2 = radiuses[1] * math.tan(beta/2) * math.cos(delta[2])
    x_S2 = x_A2 - delta_x_A2S2
    y_S2 = x_S2 * math.tan(delta[1]) + (R * fi_A1 + L_AS * math.cos(delta[0])) \
        * (math.tan(delta[0]) - math.tan(delta[1]))
    L_A2S2 = radiuses[1] * math.tan(beta/2)
    delta_y_A2S2 = L_A2S2 * math.sin(delta[2])
    y_A2 = y_S2 + delta_y_A2S2
    delta_x_A2C2 = radiuses[1] * math.sin(delta[2])
    delta_y_A2C2 = radiuses[1] * math.cos(delta[2])
    x_C2 = x_A2 + delta_x_A2C2
    y_C2 = y_A2 - delta_y_A2C2
    x_B1 = x_S1 + L_AS * math.cos(delta[1])
    x_B2 = x_S2 - radiuses[1] * math.tan(beta/2) * math.cos(delta[1])

    fi = np.arange(0, sum(i for i in coils) * 2 * math.pi, point_density)
    central_curve_point_coor = np.zeros((fi.shape[0], 3))

    for i, val in enumerate(fi):
        central_curve_point_coor[i, 0] = R * math.cos(val)
        central_curve_point_coor[i, 1] = R * math.sin(val)
        if val <= x_A1/R:
            central_curve_point_coor[i, 2] = R * val * math.tan(delta[0])
            print(R * val * math.tan(delta[0]))
        elif val > x_A1/R and val <= x_B1/R:
            central_curve_point_coor[i, 2] = y_C1 - radiuses[0] \
                * math.sqrt(1 - ((val*R - x_C1)/radiuses[0])**2)
        elif val > x_B1/R and val <= x_B2/R:
            central_curve_point_coor[i, 2] = val*R*math.tan(delta[1]) \
                + (R * fi_A1 + L_AS * math.cos(delta[0])) \
                * (math.tan(delta[0]) - math.tan(delta[1]))
        elif val > x_B2/R and val <= x_A2/R:
            central_curve_point_coor[i, 2] = y_C2 + radiuses[1] \
                * math.sqrt(1 - ((x_C2 - val * R) / radiuses[1])**2)
        elif val > x_A2/R:
            central_curve_point_coor[i, 2] = val * R * math.tan(delta[2]) \
                + y_S2 - x_S2 * math.tan(delta[2])
    plt.plot(fi, central_curve_point_coor[:, 2])
    plt.show()
    return central_curve_point_coor


def gen_central_points_height_control(parameters):
    """ Calculate coordinates of spring central curve for user defined radiuses and height.

    Parameters
    ----------
    parameters : dictionary
        The input dictionary should contain following variables: D - spring diameter [mm],
        d - spring wire diameter [mm], spring_height - height of the spring [mm],
        coils - list containing coil number for each slope [-], radiuses - list containing
        radiuses between each slope change

    Returns
    -------
    2D numpy.array
        The array has 3 colums for x, y and z point coordinates. Each row is a separate point.

    Notes
    -----
    To be added if neccessary.

    References
    ----------
    Add reference to article when published.
    """
    D = parameters['D']
    R = D/2
    d = parameters['d']
    spring_height = parameters['spring_height']
    coils = parameters['coils']
    radiuses = parameters['radiuses']
    point_density = parameters['point_density']
    delta = []
    delta.append(math.atan(d/(math.pi * D)))
    one_active_coil_pitch = (spring_height - 2 * d) / coils[1]
    delta.append(math.atan(one_active_coil_pitch/(math.pi * D)))
    delta.append(math.atan(d/(math.pi * D)))
    fi_A1 = 2 * math.pi * coils[0]
    x_A1 = R * fi_A1
    x_C1 = R * fi_A1 - radiuses[0] * math.sin(delta[0])
    z_C1 = R * fi_A1 * math.tan(delta[0]) + radiuses[0] * math.cos(delta[0])
    x_O2 = 2 * math.pi * R * sum(i for i in coils)
    z_O2 = 2 * math.pi * R * ((coils[0] + coils[2]) * math.tan(delta[0]) +
                              coils[1] * math.tan(delta[1]))
    x_A2 = x_O2 - 2 * math.pi * R * coils[2]
    z_A2 = x_A2 * math.tan(delta[0]) + z_O2 - x_O2 * math.tan(delta[0])
    x_C2 = x_A2 + radiuses[1] * math.sin(delta[2])
    z_C2 = z_A2 - radiuses[1] * math.cos(delta[2])
    delta_2C = math.atan((z_C1 - z_C2) / (x_C2 - x_C1))
    delta_2m = math.atan((radiuses[0] + radiuses[1]) /
                         math.sqrt((x_C2 - x_C1)**2 + (z_C1 - z_C2)**2
                                   - (radiuses[0] + radiuses[1])**2))
    delta_2 = delta_2m - delta_2C
    x_m1 = x_C1 + radiuses[0] * math.sin(delta_2)
    z_m1 = z_C1 - radiuses[0] * math.cos(delta_2)
    beta = abs(delta_2 - delta[0])
    L_AS = radiuses[0] * math.tan(beta/2)
    x_A1 = fi_A1 * R
    x_S1 = R * fi_A1 + L_AS * math.cos(delta[0])
    x_A2 = x_A1 + 2 * math.pi * coils[1] * R
    delta_x_A2S2 = radiuses[1] * math.tan(beta/2) * math.cos(delta[2])
    x_S2 = x_A2 - delta_x_A2S2
    x_B1 = x_S1 + L_AS * math.cos(delta[1])
    x_B2 = x_S2 - radiuses[1] * math.tan(beta/2) * math.cos(delta_2)
    
    fi = np.arange(0, sum(i for i in coils) * 2 * math.pi, point_density)
    central_curve_point_coor = np.zeros((fi.shape[0], 3))

    for i, val in enumerate(fi):
        central_curve_point_coor[i, 0] = D * math.cos(val)
        central_curve_point_coor[i, 1] = D * math.sin(val)
        if val <= x_A1/R:
            central_curve_point_coor[i, 2] = R * val * math.tan(delta[0])
        elif val > x_A1/R and val <= x_B1/R:
            central_curve_point_coor[i, 2] = z_C1 - radiuses[0] \
                * math.sqrt(1 - ((val*R - x_C1)/radiuses[0])**2)
        elif val > x_B1/R and val <= x_B2/R:
            central_curve_point_coor[i, 2] = val*R*math.tan(delta_2) \
                + z_m1 - x_m1 * math.tan(delta_2)
        elif val > x_B2/R and val <= x_A2/R:
            central_curve_point_coor[i, 2] = z_C2 + radiuses[1] \
                * math.sqrt(1 - ((x_C2 - val * R) / radiuses[1])**2)
        elif val > x_A2/R:
            central_curve_point_coor[i, 2] = val * R * math.tan(delta[2]) \
                + z_O2 - x_O2 * math.tan(delta[2])
    plt.plot(fi, central_curve_point_coor[:, 2])
    plt.show()
    return central_curve_point_coor


def gen_SpaceClaim_sript(central_points):
    """ Generate SpaceClaim input script and save it to file.

    Parameters
    ----------
    central_points : numpy.array
        Two dimensional array containing x, y, z coordinates of spring central line (helix)

    Returns
    -------
    None
        Function saves a file with the script.

    Notes
    -----
    To be added if neccessary.

    References
    ----------
    Add reference to article when published.
 """
    if algo_type == 'ControlRadius':
        wire_radius = param_control_radius['d']/2
    elif algo_type == 'ControlHeight':
        wire_radius = param_control_height['d']/2
    else:
        raise NameError('Algorithm type was not given or it was given incorrectly \
        Check variable algo_type')
    file = open(file_store_dir + 'SpaceClaim_spring_geo.py', 'w')
    for i, point in enumerate(central_points):
        file.write('Point_' + str(i+1) + '=Point.Create(MM(' + str(point[0]) + '), MM('
                   + str(point[1]) + '), MM(' + str(point[2]) + ')) \n')
        file.write('result = DatumPointCreator.Create(Point_' + str(i+1) + ')\n')
    first_run = True
    for i in range(central_points.shape[0]):
        if first_run:
            first_run = False
            file.write('selected = Selection.Create([GetRootPart().DatumPoints[' + str(i)
                       + '],')
        else:
            file.write('GetRootPart().DatumPoints[' + str(i) + '],')
    file.write(']) \n')
    file.write('result = SketchNurbs.Create(selected) \n')
    point_1 = np.array([central_points[0, 0], central_points[0, 1], central_points[0, 2]])
    point_2 = np.array([central_points[1, 0], central_points[1, 1], central_points[1, 2]])
    norm_vect = point_2 - point_1
    file.write('vector = Direction.Create(' + str(norm_vect[0]) + ', ' + str(norm_vect[1])
               + ', ' + str(norm_vect[2]) + ')\n')
    file.write('result = DatumPlaneCreator.Create(Point_1, vector, False, None)\n')
    file.write('selection = Selection.Create(GetRootPart().DatumPlanes[0])\n')
    file.write('result = ViewHelper.SetSketchPlane(selection, None)\n')
    file.write('origin = Point2D.Create(MM(0), MM(0))\n')
    file.write('result = SketchCircle.Create(origin, MM(' + str(wire_radius) + '))\n')
    file.write('selection = Selection.Create(GetRootPart().Curves[1])\n')
    file.write('secondarySelection = Selection()\n')
    file.write('options = FillOptions()\n')
    file.write('result = Fill.Execute(selection, secondarySelection, options,'
               + ' FillMode.Sketch, None)\n')
    file.write('selection = Selection.Create(GetRootPart().Bodies[0].Faces[0])\n')
    file.write('trajectories = Selection.Create(GetRootPart().Curves[0])\n')
    file.write('options = SweepFaceCommandOptions()\n')
    file.write('options.ExtrudeType = ExtrudeType.Add\n')
    file.write('options.Select = True\n')
    file.write('result = SweepFaces.Execute(selection, trajectories, options, None)\n')
    file.close()


def normalize_vect(vector):
    """Takes wector stored in numpy array and returns versor along its direction

    Parameters
    ----------
    vector : 1D numpy.array
        One dimensional array containing vector coordinates (x, y, z)

    Returns
    -------
    1D numpy.array
        Returned array contains coordinates of normalized vector
"""
    square_sum = 0
    for i in vector:
        square_sum += i**2
    length = math.sqrt(square_sum)
    return vector/length


def gen_Gmsh_sript(central_points):
    """ Generate Gmsh input script and save it to file.

    Parameters
    ----------
    central_points : numpy.array
        Two dimensional array containing x, y, z coordinates of spring central line (helix)

    Returns
    -------
    None
        Function saves a file with the script.

    Notes
    -----
    To be added if neccessary.

    References
    ----------
    Add reference to article when published.
 """
    if algo_type == 'ControlRadius':
        d = param_control_radius['d']
        D = param_control_radius['D']
    elif algo_type == 'ControlHeight':
        d = param_control_height['d']
        D = param_control_radius['D']
    else:
        raise NameError('Algorithm type was not given or it was given incorrectly \
        Check variable algo_type')
    p_1 = np.array([central_points[0, 0], central_points[0, 1], central_points[0, 2]])
    p_2 = np.array([central_points[1, 0], central_points[1, 1], central_points[1, 2]])
    vec_tangent = p_2 - p_1
    vec_any = np.array([1, 0, 0])
    vec_perpendicular_1 = np.cross(vec_tangent, vec_any)
    vec_perpendicular_2 = -vec_perpendicular_1
    vec_perpendicular_3 = np.cross(vec_tangent, vec_perpendicular_1)
    vec_perpendicular_4 = -vec_perpendicular_3
    vec_perpendicular_1_norm = normalize_vect(vec_perpendicular_1)
    vec_perpendicular_2_norm = normalize_vect(vec_perpendicular_2)
    vec_perpendicular_3_norm = normalize_vect(vec_perpendicular_3)
    vec_perpendicular_4_norm = normalize_vect(vec_perpendicular_4)
    p_3 = 0.5*d*vec_perpendicular_1_norm
    p_4 = 0.5*d*vec_perpendicular_2_norm
    p_5 = 0.5*d*vec_perpendicular_3_norm
    p_6 = 0.5*d*vec_perpendicular_4_norm
    vec_any = np.array([1, 0, -0.5])
    vec_perpendicular_1 = np.cross(vec_tangent, vec_any)
    vec_perpendicular_2 = -vec_perpendicular_1
    vec_perpendicular_3 = np.cross(vec_tangent, vec_perpendicular_1)
    vec_perpendicular_4 = -vec_perpendicular_3
    vec_perpendicular_1_norm = normalize_vect(vec_perpendicular_1)
    vec_perpendicular_2_norm = normalize_vect(vec_perpendicular_2)
    vec_perpendicular_3_norm = normalize_vect(vec_perpendicular_3)
    vec_perpendicular_4_norm = normalize_vect(vec_perpendicular_4)
    p_7 = 0.5*d*vec_perpendicular_1_norm
    p_8 = 0.5*d*vec_perpendicular_2_norm
    p_9 = 0.5*d*vec_perpendicular_3_norm
    p_10 = 0.5*d*vec_perpendicular_4_norm
    vec_any = np.array([1, 0, -1.0])
    vec_perpendicular_1 = np.cross(vec_tangent, vec_any)
    vec_perpendicular_2 = -vec_perpendicular_1
    vec_perpendicular_3 = np.cross(vec_tangent, vec_perpendicular_1)
    vec_perpendicular_4 = -vec_perpendicular_3
    vec_perpendicular_1_norm = normalize_vect(vec_perpendicular_1)
    vec_perpendicular_2_norm = normalize_vect(vec_perpendicular_2)
    vec_perpendicular_3_norm = normalize_vect(vec_perpendicular_3)
    vec_perpendicular_4_norm = normalize_vect(vec_perpendicular_4)
    p_11 = 0.5*d*vec_perpendicular_1_norm
    p_12 = 0.5*d*vec_perpendicular_2_norm
    p_13 = 0.5*d*vec_perpendicular_3_norm
    p_14 = 0.5*d*vec_perpendicular_4_norm
    vec_any = np.array([1, 0, -1.5])
    vec_perpendicular_1 = np.cross(vec_tangent, vec_any)
    vec_perpendicular_2 = -vec_perpendicular_1
    vec_perpendicular_3 = np.cross(vec_tangent, vec_perpendicular_1)
    vec_perpendicular_4 = -vec_perpendicular_3
    vec_perpendicular_1_norm = normalize_vect(vec_perpendicular_1)
    vec_perpendicular_2_norm = normalize_vect(vec_perpendicular_2)
    vec_perpendicular_3_norm = normalize_vect(vec_perpendicular_3)
    vec_perpendicular_4_norm = normalize_vect(vec_perpendicular_4)
    p_15 = 0.5*d*vec_perpendicular_1_norm
    p_16 = 0.5*d*vec_perpendicular_2_norm
    p_17 = 0.5*d*vec_perpendicular_3_norm
    p_18 = 0.5*d*vec_perpendicular_4_norm
    file = open(file_store_dir + 'Gmsh_spring_geo.geo', 'w')
    lc = 1
    file.write('SetFactory("OpenCASCADE");\n')

    for i, val in enumerate(central_points):
        file.write('Point(' + str(i) + ') = {' + str(val[0]) + ', ' + str(val[1]) + ', '
                   + str(val[2]) + ', ' + str(lc) + '}; \n')
    line_counter = 0
    for i in range(0, central_points.shape[0] - 5, 5):
        file.write('Spline(' + str(line_counter) + ') = {')
        if i == 0:
            file.write(str(i) + ', ')
            file.write(str(i + 1) + ', ')
            file.write(str(i + 2) + ', ')
            file.write(str(i + 3) + ', ')
            file.write(str(i + 4) + '};\n')
        else:
            file.write(str(i-1) + ', ')
            file.write(str(i) + ', ')
            file.write(str(i + 1) + ', ')
            file.write(str(i + 2) + ', ')
            file.write(str(i + 3) + ', ')
            file.write(str(i + 4) + '};\n')
        line_counter += 1
    starting_point = 5*line_counter
    file.write('Spline(' + str(line_counter) + ') = {')
    for i in range(starting_point, central_points.shape[0]):
        file.write(str(i - 1) + ', ')
    file.write(str(central_points.shape[0] - 1) + '};\n')
    line_counter += 1
    point_counter = central_points.shape[0]
    file.write('Wire(' + str(1) + ') = {')
    for i in range(line_counter - 1):
        file.write(str(i) + ', ')
    file.write(str(line_counter - 1) + '};\n')
    file.write('Point(' + str(point_counter + 1) + ') = {' + str(p_3[0] + D) + ', '
               + str(p_3[1]) + ', ' + str(p_3[2]) + '};\n')
    point_circle_1 = point_counter + 1
    file.write('Point(' + str(point_counter + 2) + ') = {' + str(p_4[0] + D) + ', '
               + str(p_4[1]) + ', ' + str(p_4[2]) + '};\n')
    point_circle_2 = point_counter + 2
    file.write('Point(' + str(point_counter + 3) + ') = {' + str(p_5[0] + D) + ', '
               + str(p_5[1]) + ', ' + str(p_5[2]) + '};\n')
    point_circle_3 = point_counter + 3
    file.write('Point(' + str(point_counter + 4) + ') = {' + str(p_6[0] + D) + ', '
               + str(p_6[1]) + ', ' + str(p_6[2]) + '};\n')
    point_circle_4 = point_counter + 4
    file.write('Point(' + str(point_counter + 5) + ') = {' + str(p_7[0] + D) + ', '
               + str(p_7[1]) + ', ' + str(p_7[2]) + '};\n')
    point_circle_5 = point_counter + 5
    file.write('Point(' + str(point_counter + 6) + ') = {' + str(p_8[0] + D) + ', '
               + str(p_8[1]) + ', ' + str(p_8[2]) + '};\n')
    point_circle_6 = point_counter + 6
    file.write('Point(' + str(point_counter + 7) + ') = {' + str(p_9[0] + D) + ', '
               + str(p_9[1]) + ', ' + str(p_9[2]) + '};\n')
    point_circle_7 = point_counter + 7
    file.write('Point(' + str(point_counter + 8) + ') = {' + str(p_10[0] + D) + ', '
               + str(p_10[1]) + ', ' + str(p_10[2]) + '};\n')
    point_circle_8 = point_counter + 8
    file.write('Point(' + str(point_counter + 9) + ') = {' + str(p_11[0] + D) + ', '
               + str(p_11[1]) + ', ' + str(p_11[2]) + '};\n')
    point_circle_9 = point_counter + 9
    file.write('Point(' + str(point_counter + 10) + ') = {' + str(p_12[0] + D) + ', '
               + str(p_12[1]) + ', ' + str(p_12[2]) + '};\n')
    point_circle_10 = point_counter + 10
    file.write('Point(' + str(point_counter + 11) + ') = {' + str(p_13[0] + D) + ', '
               + str(p_13[1]) + ', ' + str(p_13[2]) + '};\n')
    point_circle_11 = point_counter + 11
    file.write('Point(' + str(point_counter + 12) + ') = {' + str(p_14[0] + D) + ', '
               + str(p_14[1]) + ', ' + str(p_14[2]) + '};\n')
    point_circle_12 = point_counter + 12
    file.write('Point(' + str(point_counter + 13) + ') = {' + str(p_15[0] + D) + ', '
               + str(p_15[1]) + ', ' + str(p_15[2]) + '};\n')
    point_circle_13 = point_counter + 13
    file.write('Point(' + str(point_counter + 14) + ') = {' + str(p_16[0] + D) + ', '
               + str(p_16[1]) + ', ' + str(p_16[2]) + '};\n')
    point_circle_14 = point_counter + 14
    file.write('Point(' + str(point_counter + 15) + ') = {' + str(p_17[0] + D) + ', '
               + str(p_17[1]) + ', ' + str(p_17[2]) + '};\n')
    point_circle_15 = point_counter + 15
    file.write('Point(' + str(point_counter + 16) + ') = {' + str(p_18[0] + D) + ', '
               + str(p_18[1]) + ', ' + str(p_18[2]) + '};\n')
    point_circle_16 = point_counter + 16
    point_counter += 17
    file.write('Circle(' + str(line_counter + 1) + ') = {' + str(point_circle_2) + ', 0, ' +
               str(point_circle_6) + '};\n')
    file.write('Circle(' + str(line_counter + 2) + ') = {' + str(point_circle_6) + ', 0, ' +
               str(point_circle_10) + '};\n')
    file.write('Circle(' + str(line_counter + 3) + ') = {' + str(point_circle_10) + ', 0, ' +
               str(point_circle_14) + '};\n')
    file.write('Circle(' + str(line_counter + 4) + ') = {' + str(point_circle_14) + ', 0, ' +
               str(point_circle_4) + '};\n')
    file.write('Circle(' + str(line_counter + 5) + ') = {' + str(point_circle_4) + ', 0, ' +
               str(point_circle_8) + '};\n')
    file.write('Circle(' + str(line_counter + 6) + ') = {' + str(point_circle_8) + ', 0, ' +
               str(point_circle_12) + '};\n')
    file.write('Circle(' + str(line_counter + 7) + ') = {' + str(point_circle_12) + ', 0, ' +
               str(point_circle_16) + '};\n')
    file.write('Circle(' + str(line_counter + 8) + ') = {' + str(point_circle_16) + ', 0, ' +
               str(point_circle_1) + '};\n')
    file.write('Circle(' + str(line_counter + 9) + ') = {' + str(point_circle_1) + ', 0, ' +
               str(point_circle_5) + '};\n')
    file.write('Circle(' + str(line_counter + 10) + ') = {' + str(point_circle_5) + ', 0, ' +
               str(point_circle_9) + '};\n')
    file.write('Circle(' + str(line_counter + 11) + ') = {' + str(point_circle_9) + ', 0, ' +
               str(point_circle_13) + '};\n')
    file.write('Circle(' + str(line_counter + 12) + ') = {' + str(point_circle_13) + ', 0, ' +
               str(point_circle_3) + '};\n')
    file.write('Circle(' + str(line_counter + 13) + ') = {' + str(point_circle_3) + ', 0, ' +
               str(point_circle_7) + '};\n')
    file.write('Circle(' + str(line_counter + 14) + ') = {' + str(point_circle_7) + ', 0, ' +
               str(point_circle_11) + '};\n')
    file.write('Circle(' + str(line_counter + 15) + ') = {' + str(point_circle_11) + ', 0, ' +
               str(point_circle_15) + '};\n')
    file.write('Circle(' + str(line_counter + 16) + ') = {' + str(point_circle_15) + ', 0, ' +
               str(point_circle_2) + '};\n')
    file.write('Curve Loop(0) = {' + str(line_counter + 1) + ', ' + str(line_counter + 2)
               + ', ' + str(line_counter + 3) + ', ' + str(line_counter + 4) + ', '
               + str(line_counter + 5) + ', ' + str(line_counter + 6) + ', '
               + str(line_counter + 7) + ', ' + str(line_counter + 8)
               + ', ' + str(line_counter + 9) + ', ' + str(line_counter + 10) + ', ' +
               str(line_counter + 11) + ', ' + str(line_counter + 12) + ', '
               + str(line_counter + 13) + ', ' + str(line_counter + 14) + ', '
               + str(line_counter + 15) + ', ' + str(line_counter + 16) + '};\n')
    line_counter += 17
    file.write('Plane Surface(0) = {0};\n')
    file.write('Extrude {Surface{0};} Using Wire {1}\n')
    file.close()


if __name__ == '__main__':
    if algo_type == 'ControlRadius':
        central_points = gen_central_points_radius_control(param_control_radius)
    elif algo_type == 'ControlHeight':
        central_points = gen_central_points_height_control(param_control_height)
    else:
        raise NameError('Algorithm type was not given or it was given incorrectly\
        Check variable algo_type')
    if geom_output == 'Gmsh':
        gen_Gmsh_sript(central_points)
    elif geom_output == 'SpaceClaim':
        gen_SpaceClaim_sript(central_points)
    else:
        raise NameError('Check variable geom_output. There are two options for\
        for selection: Gmsh and SpaceClaim')
