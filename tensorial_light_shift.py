import numpy as np
import matplotlib.pyplot as plt
import re
from sympy.physics.wigner import wigner_6j
from scipy.constants import h, hbar, c, epsilon_0
from scipy.constants import physical_constants
from sympy.physics.quantum import cg

a0 = physical_constants["Bohr radius"][0]

def nm2rads(x):
    return 2*np.pi*c*1e9/x

def rads2nm(x):
    return 2*np.pi*c*1e9/x

#%% Load the transitions from txt file

def parse(s):
    try: 
        return float(s)
    except ValueError:
        x = re.search("[0-9.]*\*Pi", str(s))
        return float(x.group(0)[:-3]) * np.pi    
    
# Transitions are stored in an array w/ columns J', f (Hz), Gamma (Hz) 
transitions = np.loadtxt("transitions.txt", converters = {i: parse for i in range(3)})
transition626 = transitions[23] 

#%% Compute the scalar, vectorial and tensorial polarizibilities

J = 8
I = 5/2
F = 21/2

ELsquared = 1
eL = np.array([1, 0, 0])
wL = rads2nm(628)


def dJJprime_squared(transition, wL):
    return 3 * np.pi * epsilon_0 * hbar * c**3 * transition[2] / transition[1]**3

def alphaRWA(K, wL):
    temp_sum = np.sum([(-1)**transition[0] * (2*transition[0]+1) * float(wigner_6j(1, K, 1, J, transition[0], J)) * dJJprime_squared(transition, wL) * 1/hbar * 1/(transition[1] - wL) for transition in [transition626]])#[t for t in transitions if t[2]/2/np.pi > 0.1e6]+[transition626]])
    return (-1)**(K+J+1) * np.sqrt(2*K+1) * temp_sum

def alpha(K, wL):
    temp_sum = np.sum([(-1)**transition[0] * (2*transition[0]+1) * float(wigner_6j(1, K, 1, J, transition[0], J)) * dJJprime_squared(transition, wL) * 1/hbar * (1/(transition[1] - wL - 1j*transition[2]/2) + (-1)**K/(transition[1] + wL - 1j*transition[2]/2)).real for transition in transitions])
    return (-1)**(K+J+1) * np.sqrt(2*K+1) * temp_sum

def alphaS(wL, RWA=False):
    if RWA:
        return np.sqrt(1/(3*(2*J+1))) * alphaRWA(0, wL)
    else:
        return np.sqrt(1/(3*(2*J+1))) * alpha(0, wL)

def alphaV(wL, RWA=False):
    if RWA:
        return (-1)**(J+I+F) * np.sqrt(2*F*(2*F+1) / (F+1)) * float(wigner_6j(F, 1, F, J, I, J)) * alphaRWA(1, wL)
    else:
        return (-1)**(J+I+F) * np.sqrt(2*F*(2*F+1) / (F+1)) * float(wigner_6j(F, 1, F, J, I, J)) * alpha(1, wL)

def alphaT(wL, RWA=False):
    if RWA:
        return (-1)**(J+I+F+1) * np.sqrt(2*F*(2*F-1)*(2*F+1) / (3*(F+1)*(2*F+3))) * float(wigner_6j(F, 2, F, J, I, J)) * alphaRWA(2, wL)
    else:
        return (-1)**(J+I+F+1) * np.sqrt(2*F*(2*F-1)*(2*F+1) / (3*(F+1)*(2*F+3))) * float(wigner_6j(F, 2, F, J, I, J)) * alpha(2, wL)
    
#%% Plot polarizabilities

if __name__=="__main__":
    wArray = np.linspace(nm2rads(400), nm2rads(1000), 1000)
    alphaSarray = [alphaS(w) for w in wArray]
    alphaVarray = [alphaV(w) for w in wArray]
    alphaTarray = [alphaT(w) for w in wArray]


#%%

if __name__=="__main__":
    plt.subplot(311)
    plt.plot(2*np.pi*c/wArray*1e9, np.array(alphaSarray)/1.648e-41)
    # plt.title()
    plt.ylim(0, 700)
    
    plt.subplot(312)
    plt.plot(2*np.pi*c/wArray*1e9, np.array(alphaVarray)/1.648e-41)
    plt.ylim(-100, 100)
    
    plt.subplot(313)
    plt.plot(2*np.pi*c/wArray*1e9, np.array(alphaTarray)/1.648e-41)
    plt.ylim(-100, 100)

#%% Plot polarizabilities

if __name__=="__main__":
    wArray = np.linspace(nm2rads(610), nm2rads(640.01), 1000)
    alphaSarray = [alphaS(w, RWA=True) for w in wArray]
    alphaVarray = [alphaV(w, RWA=True) for w in wArray]
    alphaTarray = [alphaT(w, RWA=True) for w in wArray]


#%%

if __name__=="__main__":
    plt.figure()
    plt.plot(2*np.pi*c/wArray*1e9,  np.array(alphaSarray)/1.648e-41, label="Scalar")
    plt.plot(2*np.pi*c/wArray*1e9, np.array(alphaVarray)/1.648e-41, label="Vectorial")
    plt.plot(2*np.pi*c/wArray*1e9, np.array(alphaTarray)/1.648e-41, label="Tensorial")
    plt.legend()
    plt.ylim(-400, 400)

#%% Light shifts in the F, mF basis (exact)

def V_scal_F(ELsquared, wL):
    return alphaS(wL) * ELsquared

def V_vect_F(ELsquared, eL, wL, F, mF):
    cross_product = np.cross(np.conj(eL), eL)
    dot_product = cross_product[2]*mF
    return alphaV(wL) * ELsquared * dot_product/(2*F)

def V_tens_F(ELsquared, eL, wL, F, mF):
    eLstar_F = np.conj(eL)[2]*mF
    eL_F = eL[2]*mF
    Fsquared = mF**2
    return alphaT(wL) * ELsquared * (3*(eLstar_F*eL_F + eL_F*eLstar_F) - 2*Fsquared)/(2*F*(2*F-1))
    
def V_F(ELsquared, eL, wL, F, mF):
    return V_scal_F(ELsquared, eL) + V_vect_F(ELsquared, eL, wL, F, mF) + V_tens_F(ELsquared, eL, wL, F, mF)

#%% Light shifts in the I, J, mI, mJ basis (first order perturbation)

def V_IJ(ELsquared, wL, I, J, mI, mJ):
    temp_sum = 0
    for F in np.arange(np.abs(I-J), I+J+1, 1):
        for mF in np.arange(-F, F+1, 1):
            CG_coeff = complex(cg.CG(I, mI, J, mJ, F, mF).doit())
            temp_sum += CG_coeff.conjugate()*CG_coeff * V_F(ELsquared, eL, wL, F, mF)
    return temp_sum
