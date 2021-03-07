import numpy


def get_matrix_A(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122, M_1133, M_2233,
                 M_1313, ro_0):
    J = F_ff * (F_rr * F_zz - F_rz * F_zr)

    K1 = F_ff * (F_rr * T_rr + F_rz * T_rz)

    K2 = F_ff * (F_rr * T_rz + F_rz * T_zz)

    Kffrr = (F_rr * F_zz - F_rz * F_zr) * (F_rr * T_rr + F_rz * T_rz)

    Kffzr = (F_rr * F_zz - F_rz * F_zr) * (F_rr * T_rz + F_rz * T_zz)

    K3 = F_ff * (F_zr * T_rr + F_zz * T_rz)

    K4 = F_ff * (F_zr * T_rz + F_zz * T_zz)

    Kffrz = (F_zr * T_rr + F_zz * T_rz) * (F_rr * F_zz - F_rz * F_zr)

    Kffzz = (F_zr * T_rz + F_zz * T_zz) * (F_rr * F_zz - F_rz * F_zr)

    p_A = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, -(F_zz * K1 + T_rr * J) / ro_0,
            -F_rr * K1 / ro_0,
            (F_zr * K1 - T_rz * J) / ro_0,
            F_rz * K1 / ro_0,
            -Kffrr / ro_0,
            -F_rr * J / ro_0,
            0,
            -F_rz * J / ro_0,
            0],
           [0, 0, 0, 0, -(F_zz * K2 + T_rz * J) / ro_0,
            -F_rr * K2 / ro_0,
            (F_zr * K2 - T_zz * J) / ro_0,
            F_rz * K2 / ro_0,
            -Kffzr / ro_0,
            0,
            -F_rz * J / ro_0,
            -F_rr * J / ro_0,
            0],
           [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rr * M_1111, -J * F_zr * M_1111, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rr * M_1133, -J * F_zr * M_1133, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rz * M_1313, -J * F_rr * M_1313, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rr * M_1122, -J * F_zr * M_1122, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    return p_A


def get_matrix_B(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122, M_1133, M_2233,
                 M_1313, ro_0):
    J = F_ff * (F_rr * F_zz - F_rz * F_zr)

    K1 = F_ff * (F_rr * T_rr + F_rz * T_rz)

    K2 = F_ff * (F_rr * T_rz + F_rz * T_zz)

    Kffrr = (F_rr * F_zz - F_rz * F_zr) * (F_rr * T_rr + F_rz * T_rz)

    Kffzr = (F_rr * F_zz - F_rz * F_zr) * (F_rr * T_rz + F_rz * T_zz)

    K3 = F_ff * (F_zr * T_rr + F_zz * T_rz)

    K4 = F_ff * (F_zr * T_rz + F_zz * T_zz)

    Kffrz = (F_zr * T_rr + F_zz * T_rz) * (F_rr * F_zz - F_rz * F_zr)

    Kffzz = (F_zr * T_rz + F_zz * T_zz) * (F_rr * F_zz - F_rz * F_zr)

    p_B = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, -F_zz * K3 / ro_0,
            -(F_rr * K3 + T_rz * J) / ro_0,
            F_zr * K3 / ro_0,
            (F_rz * K3 - T_rr * J) / ro_0,
            -Kffrz / ro_0,
            -F_zr * J / ro_0,
            0,
            -F_zz * J / ro_0,
            0],
           [0, 0, 0, 0, -F_zz * K4 / ro_0,
            -(T_zz * J + F_rr * K4) / ro_0,
            F_zr * K4 / ro_0,
            (F_rz * K4 - T_rz * J) / ro_0,
            -Kffzz / ro_0,
            0,
            -F_zz * J / ro_0,
            -F_zr * J / ro_0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rz * M_1133, -J * F_zz * M_1133, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rz * M_3333, -J * F_zz * M_3333, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_zz * M_1313, -J * F_zr * M_1313, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, -J * F_rz * M_2233, -J * F_zz * M_2233, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    return p_B


def get_vector_F(V_r, V_z, F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333,
                 M_1122, M_1133, M_2233, M_1313, ro_0, r):
    J = F_ff * (F_rr * F_zz - F_rz * F_zr)

    p_F = [V_r,
           V_z,
           J * (F_rr * T_rr + F_rz * T_rz) / (ro_0 * r) - J * T_ff * F_ff / (ro_0 * r),
           J * (F_zr * T_rr + F_zz * T_rz) / (ro_0 * r),
           0,
           0,
           0,
           0,
           V_r / r,
           J * M_1122 * F_ff * V_r / r,
           J * M_2233 * F_ff * V_r / r,
           0,
           J * M_2222 * F_ff * V_r / r]

    return p_F


def get_next_layer(i, j, r):
    u_r = U[0][i][j][0]
    u_z = U[0][i][j][1]
    V_r = U[0][i][j][2]
    V_z = U[0][i][j][3]
    F_rr = U[0][i][j][4]
    F_zz = U[0][i][j][5]
    F_rz = U[0][i][j][6]
    F_zr = U[0][i][j][7]
    F_ff = U[0][i][j][8]
    T_rr = U[0][i][j][9]
    T_zz = U[0][i][j][10]
    T_rz = U[0][i][j][11]
    T_ff = U[0][i][j][12]

    A = get_matrix_A(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                     M_1133, M_2233, M_1313, ro_0)

    B = get_matrix_B(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                     M_1133, M_2233, M_1313, ro_0)

    F = get_vector_F(V_r, V_z, F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333,
                     M_1122, M_1133, M_2233, M_1313, ro_0, r)

    if j == 0:
        Uprev_r = numpy.matrix(U[0][i][1]).transpose()
        Unext_r = numpy.matrix(U[0][i][1]).transpose()
    else:
        Uprev_r = numpy.matrix(U[0][i][j - 1]).transpose()
        Unext_r = numpy.matrix(U[0][i][j + 1]).transpose()

    Uprev_z = numpy.matrix(U[0][i - 1][j]).transpose()
    Unext_z = numpy.matrix(U[0][i + 1][j]).transpose()

    #print(U[0][i][j])

    Uk = numpy.matrix(U[0][i][j]).transpose()
    A = numpy.matrix(A)
    #try:
    #    A_abs = numpy.matrix(numpy.linalg.eig(A)[1]) * numpy.matrix(
    #        abs(numpy.diag(numpy.linalg.eig(A)[0]))) * numpy.matrix(numpy.linalg.inv(numpy.linalg.eig(A)[1]))
    #except:
    #    print('A TAR2')
    #    print(k, i, j)
    #    # print(Uprev_z.tolist(), Unext_z.tolist())
    #    # print(r)

    B = numpy.matrix(B)
    #try:
    #    B_abs = numpy.matrix(numpy.linalg.eig(B)[1]) * numpy.matrix(
    #        abs(numpy.diag(numpy.linalg.eig(B)[0]))) * numpy.matrix(numpy.linalg.inv(numpy.linalg.eig(B)[1]))
    #except:
    #    print('B TAR2')
    #    print(k, i, j)

    F = numpy.matrix(F).transpose()

    #Uk1 = Uk - (dt / dz) * B * (Unext_z - Uprev_z) / 2.0 + (dt / dz) * B_abs * (
    #        Unext_z - 2 * Uk + Uprev_z) / 2.0 - (dt / dr) * A * (Unext_r - Uprev_r) / 2.0 + (
    #             dt / dr) * A_abs * (Unext_r - 2 * Uk + Uprev_r) / 2.0 + dt * F

    Uk1 = Uk - (dt / dz) * B * (Unext_z - Uprev_z) + (dt / dr) * A * (Unext_r - Uprev_r) + dt * F


    Uk1 = Uk1.transpose()
    U[1][i][j] = Uk1.tolist()[0]

    if j == 0:
        U[1][i][j][2] = 0.0
        U[1][i][j][11] = 0.0


#------------------------------------------INITIAL-VALUES--------------------------------------------------------------
u_r = 0.0
u_z = 0.0
V_r = 0.0
V_z_Str = 300.0
V_z_Tar = 0.0
F_rr = 1.0
F_zz = 1.0
F_rz = 0.0
F_zr = 0.0
F_ff = 1.0
T_rr = 0.0
T_zz = 0.0
T_rz = 0.0
T_ff = 0.0
M_1111_Tar = 50.0e9
M_2222_Tar = 50.0e9
M_3333_Tar = 0.5e9
M_1122_Tar = 1.5e9
M_1133_Tar = 0.015e9
M_2233_Tar = 0.015e9
M_1313_Tar = 2.5e9

M_1111_Str = 300.0e9
M_2222_Str = 300.0e9
M_3333_Str = 300.0e9
M_1122_Str = 150.0e9
M_1133_Str = 150.0e9
M_2233_Str = 150.0e9
M_1313_Str = 75.0e9

ro_0_Str = 7800.0
ro_0_Tar = 1500.0


U = [[[[]]], [[[]]]]
N = 100

for k in range(2):
    for i in range(int(0.8 * N)):
        for j in range(int(0.2 * N) + 1):
            for n in range(13):
                U[k][i][j].append(0)
            if j != int(0.2 * N):
                U[k][i].append([])
        if i != int(0.8 * N):
            U[k].append([[]])

    for i in range(int(0.8 * N), N):
        for j in range(N + 1):
            for n in range(13):
                U[k][i][j].append(0)
            if j != N:
                U[k][i].append([])
        if i != N - 1:
            U[k].append([[]])

for i in range(int(0.8 * N)):
    for j in range(int(0.2 * N) + 1):
        U[0][i][j][0] = u_r
        U[0][i][j][1] = u_z
        U[0][i][j][2] = V_r
        U[0][i][j][3] = V_z_Str
        U[0][i][j][4] = F_rr
        U[0][i][j][5] = F_zz
        U[0][i][j][6] = F_rz
        U[0][i][j][7] = F_zr
        U[0][i][j][8] = F_ff
        U[0][i][j][9] = T_rr
        U[0][i][j][10] = T_zz
        U[0][i][j][11] = T_rz
        U[0][i][j][12] = T_ff

for i in range(int(0.8 * N), N):
    for j in range(N + 1):
        U[0][i][j][0] = u_r
        U[0][i][j][1] = u_z
        U[0][i][j][2] = V_r
        U[0][i][j][3] = V_z_Tar
        U[0][i][j][4] = F_rr
        U[0][i][j][5] = F_zz
        U[0][i][j][6] = F_rz
        U[0][i][j][7] = F_zr
        U[0][i][j][8] = F_ff
        U[0][i][j][9] = T_rr
        U[0][i][j][10] = T_zz
        U[0][i][j][11] = T_rz
        U[0][i][j][12] = T_ff

T = 0.000006

dr = 0.01 / float(N)
dz = 0.0125 / float(N)

kk = 1
K = 4

t = 0
Index = 1
#dt = T / float(100 * N)

while t < T:
    # --------------------------------------------------TIME-STER---------------------------------------------------------
    M_1111 = M_1111_Str
    M_2222 = M_2222_Str
    M_3333 = M_3333_Str
    M_1122 = M_1122_Str
    M_1133 = M_1133_Str
    M_2233 = M_2233_Str
    M_1313 = M_1313_Str

    ro_0 = ro_0_Str

    dt_Str = float('+inf')
    R = 0.8  # Courant's constant

    for i in range(int(0.8 * N)):
        for j in range(int(0.2 * N + 1)):
            u_r = U[0][i][j][0]
            u_z = U[0][i][j][1]
            V_r = U[0][i][j][2]
            V_z = U[0][i][j][3]
            F_rr = U[0][i][j][4]
            F_zz = U[0][i][j][5]
            F_rz = U[0][i][j][6]
            F_zr = U[0][i][j][7]
            F_ff = U[0][i][j][8]
            T_rr = U[0][i][j][9]
            T_zz = U[0][i][j][10]
            T_rz = U[0][i][j][11]
            T_ff = U[0][i][j][12]

            A = get_matrix_A(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                             M_1133, M_2233, M_1313, ro_0)

            B = get_matrix_B(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                             M_1133, M_2233, M_1313, ro_0)

            try:
                max_A_Str = max(abs(numpy.linalg.eig(A)[0]))
                max_B_Str = max(abs(numpy.linalg.eig(B)[0]))
            except:
                print('dt STR')
                print(k, i, j)
                print(U[0][i][j])

            dt_ij = R / (max_A_Str / dr + max_B_Str / dz) / 100.0

            if dt_Str > dt_ij:
                dt_Str = dt_ij
            else:
                pass

    M_1111 = M_1111_Tar
    M_2222 = M_2222_Tar
    M_3333 = M_3333_Tar
    M_1122 = M_1122_Tar
    M_1133 = M_1133_Tar
    M_2233 = M_2233_Tar
    M_1313 = M_1313_Tar

    ro_0 = ro_0_Tar

    dt_Tar = float('+inf')

    for i in range(int(0.8 * N), N):
        for j in range(N + 1):
            u_r = U[0][i][j][0]
            u_z = U[0][i][j][1]
            V_r = U[0][i][j][2]
            V_z = U[0][i][j][3]
            F_rr = U[0][i][j][4]
            F_zz = U[0][i][j][5]
            F_rz = U[0][i][j][6]
            F_zr = U[0][i][j][7]
            F_ff = U[0][i][j][8]
            T_rr = U[0][i][j][9]
            T_zz = U[0][i][j][10]
            T_rz = U[0][i][j][11]
            T_ff = U[0][i][j][12]

            A = get_matrix_A(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                             M_1133, M_2233, M_1313, ro_0)

            B = get_matrix_B(F_rr, F_rz, F_zz, F_zr, F_ff, T_rr, T_zz, T_rz, T_ff, M_1111, M_2222, M_3333, M_1122,
                             M_1133, M_2233, M_1313, ro_0)
            try:
                max_A_Str = max(abs(numpy.linalg.eig(A)[0]))
                max_B_Str = max(abs(numpy.linalg.eig(B)[0]))

            except:
                print('dt TAR')
                print(k, i, j)
                print(U[0][i][j])

            try:
                dt_ij = R / (max_A_Str / dr + max_B_Str / dz)
            except:
                print(max_A_Str, max_B_Str)

            if dt_Tar > dt_ij:
                dt_Tar = dt_ij
            else:
                pass

    dt = min(dt_Str, dt_Tar) / 100.0
    print(dt)

    # ------------------------------------------STRIKER-------------------------------------------------------------------

    M_1111 = M_1111_Str
    M_2222 = M_2222_Str
    M_3333 = M_3333_Str
    M_1122 = M_1122_Str
    M_1133 = M_1133_Str
    M_2233 = M_2233_Str
    M_1313 = M_1313_Str

    ro_0 = ro_0_Str

    for i in range(1, int(0.8 * N)):
        for j in range(int(0.2 * N)):
            r = dr
            get_next_layer(i, j, r)
            if j != 0:
                r = r + dr

        for j in range(int(0.2 * N), int(0.2 * N) + 1):
            # Uk = U[0][i][j - 1]
            #Uk = U[1][i][j - 2] * -1
            Uk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range (0, 12):
                Uk[k] = U[1][i][j - 2][k] * 1

            U[1][i][j] = Uk

            U[1][i][j][9] = 0.0
            U[1][i][j][11] = 0.0

    for j in range(0, int(0.2 * N) + 1):
        # Uk = U[0][1][j]
        Uk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for k in range(0, 12):
            Uk[k] = U[1][2][j][k]

        U[1][0][j] = Uk
        U[1][0][j][10] = 0.0
        U[1][0][j][11] = 0.0

    # -------------------------------------TARGET---------------------------------------

    M_1111 = M_1111_Tar
    M_2222 = M_2222_Tar
    M_3333 = M_3333_Tar
    M_1122 = M_1122_Tar
    M_1133 = M_1133_Tar
    M_2233 = M_2233_Tar
    M_1313 = M_1313_Tar

    ro_0 = ro_0_Tar

    r = dr
    for i in range(int(0.8 * N), int(0.8 * N) + 1):
        for j in range(int(0.2 * N) + 1):
            r = dr
            get_next_layer(i, j, r)
            if j != 0:
                r = r + dr

    for i in range(int(0.8 * N) + 1, N - 1):
        r = dr
        for j in range(N):
            r = dr
            get_next_layer(i, j, r)
            if j != 0:
                r = r + dr

        for j in range(N, N + 1):
            # Uk = U[0][i][j - 1]

            Uk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(0, 12):
                Uk[k] = U[0][i][j - 2][k] 

            U[1][i][j] = Uk
            U[1][i][j][9] = 0.0
            U[1][i][j][11] = 0.0

    for i in range(N - 1, N):
        for j in range(0, N + 1):
            # Uk = U[0][i - 1][j]

            Uk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(0, 12):
                Uk[k] = U[0][i - 2][j][k] 

            U[1][i][j] = Uk
            U[1][i][j][10] = 0.0
            U[1][i][j][11] = 0.0

    for i in range(int(0.8 * N), int(0.8 * N) + 1):
        for j in range(int(0.2 * N) + 1, N + 1):
            # Uk = U[0][i + 1][j]

            Uk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(0, 12):
                Uk[k] = U[0][i + 2][j][k]

            U[1][i][j] = Uk
            U[1][i][j][10] = 0.0
            U[1][i][j][11] = 0.0


    a_koef = 0.2

    for i in range(1, int(0.8 * N) + 1):
        for j in range(1, int(0.2 * N)):
            U1 = numpy.matrix(U[1][i][j]).transpose()
            U0 = numpy.matrix(U[0][i][j]).transpose()
            U0next_r = numpy.matrix(U[0][i + 1][j]).transpose()
            U0prev_r = numpy.matrix(U[0][i - 1][j]).transpose()

            U0next_z = numpy.matrix(U[0][i][j + 1]).transpose()
            U0prev_z = numpy.matrix(U[0][i][j - 1]).transpose()

            U_smooth = U1 + a_koef * (U0next_r + U0prev_r + U0next_z + U0prev_z - 4 * U0)
            U[1][i][j] = U_smooth.transpose().tolist()[0]

    for i in range(int(0.8 * N), int(0.8 * N)):
        for j in range(int(0.2 * N)):
            U1 = numpy.matrix(U[1][i][j]).transpose()
            U0 = numpy.matrix(U[0][i][j]).transpose()
            U0next_r = numpy.matrix(U[0][i + 1][j]).transpose()
            U0prev_r = numpy.matrix(U[0][i - 1][j]).transpose()

            U0next_z = numpy.matrix(U[0][i][j + 1]).transpose()
            U0prev_z = numpy.matrix(U[0][i][j - 1]).transpose()

            U_smooth = U1 + a_koef * (U0next_r + U0prev_r + U0next_z + U0prev_z - 4 * U0)
            U[1][i][j] = U_smooth.transpose().tolist()[0]

    for i in range(int(0.8 * N) + 1, N - 1):
        for j in range(1, N -1):
            U1 = numpy.matrix(U[1][i][j]).transpose()
            U0 = numpy.matrix(U[0][i][j]).transpose()
            U0next_r = numpy.matrix(U[0][i + 1][j]).transpose()
            U0prev_r = numpy.matrix(U[0][i - 1][j]).transpose()

            U0next_z = numpy.matrix(U[0][i][j + 1]).transpose()
            U0prev_z = numpy.matrix(U[0][i][j - 1]).transpose()

            U_smooth = U1 + a_koef * (U0next_r + U0prev_r + U0next_z + U0prev_z - 4 * U0)
            U[1][i][j] = U_smooth.transpose().tolist()[0]



    Uprev_z = numpy.matrix(U[1][int(0.8 * N) - 1][int(0.2 * N)]).transpose()


    Unext_z = numpy.matrix(U[1][int(0.8 * N) + 1][int(0.2 * N)]).transpose()

    Uprev_r = numpy.matrix(U[1][int(0.8 * N)][int(0.2 * N) - 1]).transpose()
    Unext_r = numpy.matrix(U[1][int(0.8 * N)][int(0.2 * N) + 1]).transpose()
    Uprev_z2 = numpy.matrix(U[1][int(0.8 * N) - 2][int(0.2 * N)]).transpose()
    Unext_z2 = numpy.matrix(U[1][int(0.8 * N) + 2][int(0.2 * N)]).transpose()
    Uprev_r2 = numpy.matrix(U[1][int(0.8 * N)][int(0.2 * N) - 2]).transpose()
    Unext_r2 = numpy.matrix(U[1][int(0.8 * N)][int(0.2 * N) + 2]).transpose()
    Uk1 = (Uprev_z + Unext_z + Uprev_r + Unext_r) / 4.0 #+ (Uprev_z2 + Unext_z2 + Uprev_r2 + Unext_r2) / 8.0
    Uk1 = Uk1.transpose()
    U[1][int(0.8 * N)][int(0.2 * N)] = Uk1.tolist()[0]

    t = t + dt
    Index = 1
    z = 0.0
    # if k == 2:

    Solution_File = open("C:\\Users\\useru\\Desktop\\Netgazer32_2\\data_1\\MySolution" + str(kk) + '.mv', 'w')

    Solution_File.write(str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
            N + 1) + 1) + ' ' + '3' + ' ' + '11' + ' ' + 'V_r V_z F_rr F_zz F_rz F_zr F_ff T_rr T_zz T_rz T_ff' + '\n')

    for i in range(int(0.8 * N)):
        r = 0.0
        for j in range(int(0.2 * N) + 1):
            Solution_File.write(str(Index) + ' ' + str(z) + ' ' + str(r) + ' 0 ')
            for n in range(2, 13):
                Solution_File.write(str(complex(U[0][i][j][n]).real) + ' ')
            Solution_File.write('\n')
            r += dr
            Index += 1
        z += dz

    for i in range(int(0.8 * N), N):
        r = 0.0
        for j in range(N + 1):
            Solution_File.write(str(Index) + ' ' + str(z) + ' ' + str(r) + ' 0 ')
            for n in range(2, 13):
                Solution_File.write(str(complex(U[0][i][j][n]).real) + ' ')
            Solution_File.write('\n')
            r += dr
            Index += 1
        z += dz

    Solution_File.write(str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
            N + 1) + 1) + ' 0.011 0 -0.001 0 0 0 0 0 0 0 0 0 0 0\n')

    Solution_File.write(
        str(int((0.8 * N) * (2 * 0.2 * N - 2) + (0.2 * N) * (2 * N - 2)) + 7)
        + ' 3 1 U\n')

    Index = 1
    Vertex = 1
    for i in range(int(0.8 * N)):
        for j in range(0, int(0.2 * N)):
            Solution_File.write(
                str(Index) + ' ' + str(Vertex) + ' ' + str(Vertex + 1) + ' ' + str(Vertex + int(0.2 * N) + 2) + ' 0\n')
            Index += 1
            Solution_File.write(
                str(Index) + ' ' + str(Vertex) + ' ' + str(Vertex + int(0.2 * N) + 1) + ' ' + str(
                    Vertex + int(0.2 * N) + 2) + ' 0\n')
            Vertex += 1
            Index += 1
        Vertex += 1

    Solution_File.write(
        str(Index) + ' 1 ' + str(int(0.8 * N * (0.2 * N + 1)) + 1) + ' ' + str(
            int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                    N + 1) + 1) + ' 0\n')

    Index += 1
    Solution_File.write(
        str(Index) + ' 1 ' + str(int(0.2 * N + 1)) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                N + 1) + 1) + ' 0\n')

    Index += 1
    Solution_File.write(
        str(Index) + ' ' + str(int(0.2 * N + 1)) + ' ' + str(int(0.8 * N * (0.2 * N + 1) + 0.2 * N + 1)) + ' ' + str(
            int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                    N + 1) + 1) + ' 0\n')

    Index += 1

    for i in range(int(0.2 * N) - 1):
        for j in range(0, N):
            Solution_File.write(
                str(Index) + ' ' + str(Vertex) + ' ' + str(Vertex + 1) + ' ' + str(Vertex + N + 2) + ' 0\n')
            Index += 1
            Solution_File.write(
                str(Index) + ' ' + str(Vertex) + ' ' + str(Vertex + N + 1) + ' ' + str(Vertex + N + 2) + ' 0\n')
            Vertex += 1
            Index += 1
        Vertex += 1

    Solution_File.write(
        str(Index) + ' ' + str(int(0.8 * N * (0.2 * N + 1) + N + 1)) + ' ' + str(
            int(0.8 * N * (0.2 * N + 1) + 0.2 * N + 1)) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                N + 1) + 1) + ' 0\n')
    Index += 1

    Solution_File.write(
        str(Index) + ' ' + str(int(0.8 * N * (0.2 * N + 1) + N + 1)) + ' ' + str(
            int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                    N + 1)) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                N + 1) + 1) + ' 0\n')

    Index += 1

    Solution_File.write(
        str(Index) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N - 1) * (
                N + 1) + 1) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                N + 1)) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                N + 1) + 1) + ' 0\n')

    Index += 1

    Solution_File.write(
        str(Index) + ' ' + str(int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N - 1) * (
                N + 1) + 1) + ' ' + str(int(0.8 * N * (0.2 * N + 1)) + 1) + ' ' + str(
            int(0.2 * N + 1) * int(0.8 * N) + int(0.2 * N) * (
                    N + 1) + 1) + ' 0\n')

    for i in range(int(0.8 * N)):
        for j in range(int(0.2 * N) + 1):
            U[0][i][j] = U[1][i][j]

    for i in range(int(0.8 * N), N):
        for j in range(N + 1):
            U[0][i][j] = U[1][i][j]

    print('recorded ' + str(kk))

    kk = kk + 1
