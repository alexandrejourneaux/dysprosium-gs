import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import physical_constants
from scipy.constants import h
from tikzplotlib import clean_figure as tikz_clean, save as tikz_save

from tensorial_light_shift import V_IJ, transition626, nm2rads, rads2nm

no_light_shift = True

muB, muN = physical_constants["Bohr magneton"][0], physical_constants["nuclear magneton"][0]

J = 8
I = 5/2

dimI = int(2*I + 1)
dimJ = int(2*J + 1)
dim_space = dimJ * dimI

mIs = np.linspace(-I, I, int(2*I+1))
mJs = np.linspace(-J, J, int(2*J+1))

Se = 2
Le = 6
gJ = 1 + (J*(J+1) + Se*(Se+1) - Le*(Le+1)) / (2*J*(J+1))

Sn = 5/2
Ln = 0
gI = 1 + (J*(J+1) + Sn*(Sn+1) - Ln*(Ln+1)) / (2*J*(J+1))

def index(proj, norm):
    """Given the quantum numbers mJ and J (or mI and I), returns the index of 
    mJ in the ordered list of authorized mJ’s"""
    return np.where(np.linspace(-norm, norm, int(2*norm+1)) == proj)[0][0] 

def state_num(mI, mJ):
    return(dimJ * index(mI, I) + index(mJ, J))

def state(mI, mJ):
    """State |mI, mJ> is a 1 at position dimJ * index(mI, I) + index(mJ, J)"""
    res = np.zeros((dim_space,))
    res[state_num(mI, mJ)] = 1
    return res

Identity = np.eye(dim_space)

#%% Definition of I operators 

Iplus = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        if mI+1 in mIs:
            Iplus[state_num(mI+1, mJ), state_num(mI, mJ)] = np.sqrt(I*(I+1) - mI*(mI+1))

Iminus = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        if mI-1 in mIs:
            Iminus[state_num(mI-1, mJ), state_num(mI, mJ)] = np.sqrt(I*(I+1) - mI*(mI-1))        

Iz = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        Iz[state_num(mI, mJ), state_num(mI, mJ)] = mI
            
Ix = 1/2 * (Iplus + Iminus)
Iy = 1/2j * (Iplus - Iminus)



#%% Definition of J operators

Jplus = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        if mJ+1 in mJs:
            Jplus[state_num(mI, mJ+1), state_num(mI, mJ)] = np.sqrt(J*(J+1) - mJ*(mJ+1))

Jminus = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        if mJ-1 in mJs:
            Jminus[state_num(mI, mJ-1), state_num(mI, mJ)] = np.sqrt(J*(J+1) - mJ*(mJ-1))


Jz = np.zeros((dim_space, dim_space))
for mI in mIs:
    for mJ in mJs:
        Jz[state_num(mI, mJ), state_num(mI, mJ)] = mJ
            
Jx = 1/2 * (Jplus + Jminus)
Jy = 1/2j * (Jplus - Jminus)

#%% Diagonalization of the Hamiltonian

A = -116.231e6
B = 1091.577e6

power = 4 # W
waist = 50e-6 # m
detuning = 100e6
wavelength = rads2nm(transition626[1])*1e9

IdotJ = Ix@Jx + Iy@Jy + Iz@Jz

H = lambda magBgauss: (A * IdotJ
                    + B * (3/2 * IdotJ@(2*IdotJ + 1) - I*(I+1)*J*(J+1) * Identity) / (2*I*(2*I-1)*J*(2*J-1))
                    + gJ * muB * magBgauss*1e-4 * Jz / h
                    + gI * muN *  magBgauss*1e-4 * Iz / h)

if no_light_shift:
    b_array = np.linspace(0, 3000, 200)
    eigvals = np.empty((dim_space, len(b_array)))
    for j, b in enumerate(b_array):
        eigvals[:, j] = np.linalg.eigh(H(b))[0]
    
    plt.figure()
    for i in range(dim_space):    
        plt.plot(b_array, eigvals[i]/1e9)
    
    plt.xlim((0, 2000))
    plt.ylim((-8, 8))
    plt.xlabel("Magnetic field ($G$)")
    plt.ylabel("Energy ($GHz$)")
    # plt.savefig("eigenenergies_vs_mag_field.png")
    #tikz_clean()
    #tikz_save("zeeman_paschen_back.tikz")
    
else:
    b_array = np.linspace(0, 3000, 200)
    eigvals_unperturbed = np.empty((dim_space, len(b_array)))
    eigvects = np.empty((dim_space, dim_space, len(b_array)), dtype=complex)
    for j, b in enumerate(b_array):
        eigvals_unperturbed[:, j], eigvects[:, :, j] = np.linalg.eigh(H(b))
    
    
    V_IJ_array = []
    for mI in mIs:
        for mJ in mJs:
            print(mI, mJ)
            V_IJ_array.append(V_IJ(power, waist, wavelength, mI, mJ))
    
    eigvals_perturbed = np.empty(eigvals_unperturbed.shape)
    for i in tqdm(enumerate(range(dim_space))):
        for j in range(len(b_array)):
            eigvals_perturbed[i,j] = eigvals_unperturbed[i,j] + np.sum([eigvects[u,i,j].conjugate()*eigvects[u,i,j]*V_IJ_array[u] for u in range(len(V_IJ_array))])/h
    
    plt.figure()
    for i in range(dim_space):    
        plt.plot(b_array, eigvals_perturbed[i])
    
    plt.xlim((0, 2000))
    plt.ylim((-8e9, 8e9))
    plt.xlabel("Magnetic field (Gauss)")
    plt.ylabel("Energy (GHz)")
