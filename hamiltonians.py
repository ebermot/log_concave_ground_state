"""Quantum Hamiltonians for various physical systems and annealing protocols.

This module provides functions to construct Hamiltonians for quantum annealing,
particularly focused on independent set problems and bipartite systems.
"""

from typing import Dict, Callable
import numpy as np
import qutip as qt 
from itertools import product 
from helpers import sigx

# --- Basic Hamiltonians ---

def h0(n: int) -> qt.Qobj:
    """Construct the initial transverse field Hamiltonian for quantum annealing.
    
    Args:
        n: Number of qubits
        
    Returns:
        Initial Hamiltonian as sum of Pauli X operators
    """
    return -sum(sigx(n))

def h0_bipartite(n: int) -> qt.Qobj:
    """Construct the initial transverse field Hamiltonian for quantum annealing for bipartite systems.
    
    Args:
        n: Number of qubits per partition
        
    Returns:
        Initial bipartite Hamiltonian
    """
    identity_tensor = qt.tensor([qt.qeye(2)]*n)
    return qt.tensor(identity_tensor, h0(n)) + qt.tensor(h0(n), identity_tensor)

def h0_hw(n: int) -> qt.Qobj:
    """Construct the initial transverse field Hamiltonian in Hamming weight basis.
    
    Args:
        n: System size
        
    Returns:
        Initial Hamiltonian in Hamming weight basis
    """
    mat = np.zeros((n+1, n+1))
    for d in range(n):
        a_d = np.sqrt((d+1) * (n-d))  
        mat[d, d+1] = mat[d+1, d] = -a_d
    return qt.Qobj(mat, dims=[[n+1], [n+1]])

def h0_bipartite_hw(n: int) -> qt.Qobj:
    """Construct the initial transverse field Hamiltonian for bipartite systems in Hamming weight basis.
    
    Args:
        n: System size
        
    Returns:
        Initial bipartite Hamiltonian in Hamming weight basis
    """
    return qt.tensor(qt.qeye(n+1), h0_hw(n)) + qt.tensor(h0_hw(n), qt.qeye(n+1))

# --- Problem Hamiltonians ---

def h1(f: Callable, args: Dict) -> qt.Qobj:
    """Construct the problem Hamiltonian from a cost function.
    
    Args:
        f: Cost function taking bitstring and args
        args: Dictionary containing problem parameters
        
    Returns:
        Problem Hamiltonian
    """
    n = args['n']
    basis = [[int(bit) for bit in bitstring] for bitstring in product('01', repeat=n)]
    diag = [f(b, args) for b in basis]
    return qt.Qobj(np.diag(diag), dims=[[2]*n, [2]*n])

def h1_hw(f: Callable, args: Dict) -> qt.Qobj:
    """Construct the problem Hamiltonian in Hamming weight basis.
    
    Args:
        f: Cost function taking Hamming weight and args
        args: Dictionary containing problem parameters
        
    Returns:
        Problem Hamiltonian in Hamming weight basis
    """
    n = args['n']
    basis = np.arange(0, n+1)
    diag = [f(b, args) for b in basis]
    return qt.Qobj(np.diag(diag), dims=[[n+1], [n+1]])


def h1_bipartite(f: Callable, args: Dict) -> qt.Qobj:
    """Construct the bipartite problem Hamiltonian.
    
    Args:
        f: Cost function taking left/right partition states and args
        args: Dictionary containing problem parameters
        
    Returns:
        Bipartite problem Hamiltonian
    """
    n = args['n']
    if args['balanced']:
        nL = nR = n
        gamma = 1
    else:
        nL = n
        gamma = args['gamma']
        nR = int(gamma*nL)
    
    basis = [[int(bit) for bit in bitstring] for bitstring in product('01', repeat=nL+nR)]
    diag = []
    for b in basis:
        left, right = b[:nL], b[nL:]
        x = int("".join(map(str, left)), 2)
        y = int("".join(map(str, right)), 2)
        diag.append(f(x, y, args))
    return qt.Qobj(np.diag(diag), dims=[[2]*(nL+nR), [2]*(nL+nR)])

def h1_bipartite_hw(f: Callable, args: Dict) -> qt.Qobj:
    """Construct the problem Hamiltonian for bipartite systems in Hamming weight basis.
    
    Args:
        f: Cost function taking left/right Hamming weights and args
        args: Dictionary containing problem parameters
        
    Returns:
        Problem Hamiltonian for bipartite systems in Hamming weight basis
    """
    n = args['n']
    if args['balanced']:
        nL = nR = n
        gamma = 1
    else:
        nL = n
        gamma = args['gamma']
        nR = int(gamma*nL)
    
    basis_x = np.arange(0, nL+1)
    basis_y = np.arange(0, nR+1)
    diag = np.zeros(((nL+1)*(nR+1),))
    
    for b_x in basis_x:
        for b_y in basis_y:
            index = b_x * (nL+1) + b_y
            diag[index] = f(b_x, b_y, args)
    
    return qt.Qobj(np.diag(diag), dims=[[nL+1, nR+1], [nL+1, nR+1]])

# --- Annealing Schedules ---

def hs_annealing(t: float, f: Callable, args: Dict) -> qt.Qobj:
    """Construct the annealing schedule Hamiltonian.
    
    Args:
        t: Annealing parameter (0 to 1)
        f: Cost function
        args: Dictionary containing problem parameters
        
    Returns:
        Annealing Hamiltonian at time t
    """
    n = args['n']
    return (1-t)*h0(n) + t*h1(f, args)

def hs_hw_annealing(t: float, f: Callable, args: Dict) -> qt.Qobj:
    """Construct the annealing schedule Hamiltonian in Hamming weight basis.
    
    Args:
        t: Annealing parameter (0 to 1)
        f: Cost function
        args: Dictionary containing problem parameters
        
    Returns:
        Annealing Hamiltonian in Hamming weight basis at time t
    """
    n = args['n']
    return (1-t)*h0_hw(n) + t*h1_hw(f, args)

def hs_bipartite_annealing(t: float, f: Callable, args: Dict) -> qt.Qobj:
    """Construct the annealing schedule Hamiltonian for bipartite systems.
    
    Args:
        t: Annealing parameter (0 to 1)
        f: Cost function
        args: Dictionary containing problem parameters
        
    Returns:
        Bipartite annealing Hamiltonian at time t
    """
    n = args['n']
    return (1-t)*h0_bipartite(n) + t*h1_bipartite(f, args)

def hs_bipartite_hw_annealing(t: float, f: Callable, args: Dict) -> qt.Qobj:
    """Construct the annealing schedule Hamiltonian for bipartite systems in Hamming weight basis.
    
    Args:
        t: Annealing parameter (0 to 1)
        f: Cost function
        args: Dictionary containing problem parameters
        
    Returns:
        Annealing Hamiltonian for bipartite systems in Hamming weight basis at time t
    """
    n = args['n']
    return (1-t)*h0_bipartite_hw(n) + t*h1_bipartite_hw(f, args)

