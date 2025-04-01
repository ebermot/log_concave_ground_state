"""Module for defining custom functions for quantum annealing simulations.

This module provides a framework for defining custom functions that can be used
in the quantum annealing simulation framework. Each function should be defined
in both Hamming weight and bitstring bases.
"""

from typing import Dict, Any, Callable, Tuple
import numpy as np
import qutip as qt

class CustomFunction:
    """Class to define a custom function with both Hamming weight and bitstring implementations.
    
    Attributes:
        name: Name of the function
        hw_func: Function in Hamming weight basis
        bitstring_func: Function in bitstring basis
        hw_hamiltonian: Hamiltonian in Hamming weight basis
        bitstring_hamiltonian: Hamiltonian in bitstring basis
        parameters: Dictionary of function parameters
    """
    
    def __init__(
        self,
        name: str,
        hw_func: Callable[[float, Dict[str, Any]], float],
        bitstring_func: Callable[[float, Dict[str, Any]], float],
        hw_hamiltonian: Callable[[float, int, Dict[str, Any]], Any],
        bitstring_hamiltonian: Callable[[float, int, Dict[str, Any]], Any],
        parameters: Dict[str, Any]
    ):
        """Initialize a custom function.
        
        Args:
            name: Name of the function
            hw_func: Function in Hamming weight basis
            bitstring_func: Function in bitstring basis
            hw_hamiltonian: Hamiltonian in Hamming weight basis
            bitstring_hamiltonian: Hamiltonian in bitstring basis
            parameters: Dictionary of function parameters
        """
        self.name = name
        self.hw_func = hw_func
        self.bitstring_func = bitstring_func
        self.hw_hamiltonian = hw_hamiltonian
        self.bitstring_hamiltonian = bitstring_hamiltonian
        self.parameters = parameters

def example_custom_function() -> CustomFunction:
    """Create an example custom function.
    
    This is a template for creating custom functions. Replace the function
    implementations with your desired behavior.
    
    Returns:
        CustomFunction instance with example implementations
    """
    def hw_example(x: float, params: Dict[str, Any]) -> float:
        """Example function in Hamming weight basis.
        
        Args:
            x: Input value
            params: Function parameters
            
        Returns:
            Function value
        """
        n = params['n']
        a = params.get('a', 1.0)
        b = params.get('b', 0.0)
        return a * x**2 + b * x + n/4
    
    def bitstring_example(x: float, params: Dict[str, Any]) -> float:
        """Example function in bitstring basis.
        
        Args:
            x: Input value
            params: Function parameters
            
        Returns:
            Function value
        """
        n = params['n']
        a = params.get('a', 1.0)
        b = params.get('b', 0.0)
        return a * x**2 + b * x + n/4
    
    def hw_ham_example(t: float, n: int, params: Dict[str, Any]) -> Any:
        """Example Hamiltonian in Hamming weight basis.
        
        Args:
            t: Time parameter
            n: System size
            params: Function parameters
            
        Returns:
            Hamiltonian operator
        """
        # Implement your Hamiltonian here
        # This should return a qutip.Qobj
        pass
    
    def bitstring_ham_example(t: float, n: int, params: Dict[str, Any]) -> Any:
        """Example Hamiltonian in bitstring basis.
        
        Args:
            t: Time parameter
            n: System size
            params: Function parameters
            
        Returns:
            Hamiltonian operator
        """
        # Implement your Hamiltonian here
        # This should return a qutip.Qobj
        pass
    
    return CustomFunction(
        name="example",
        hw_func=hw_example,
        bitstring_func=bitstring_example,
        hw_hamiltonian=hw_ham_example,
        bitstring_hamiltonian=bitstring_ham_example,
        parameters={'a': 1.0, 'b': 0.0}
    )

def double_well_function() -> CustomFunction:
    """Create a double-well potential function.
    
    This function implements a double-well potential with:
    - Two minima at x = 0 and x = n
    - A barrier between the wells
    - Tunable barrier height and width
    
    Returns:
        CustomFunction instance with double-well implementations
    """
    def hw_double_well(x: float, params: Dict[str, Any]) -> float:
        """Double-well potential in Hamming weight basis.
        
        Args:
            x: Input value (Hamming weight)
            params: Function parameters including:
                - n: System size
                - barrier_height: Height of the barrier between wells
                - barrier_width: Width of the barrier
                
        Returns:
            Potential value
        """
        n = params['n']
        barrier_height = params.get('barrier_height', 1.0)
        barrier_width = params.get('barrier_width', 0.2)
        
        # Normalize x to [0,1]
        x_norm = x / n
        
        # Create double well potential
        # V(x) = (x - n/2)^2 + barrier_height * exp(-(x - n/2)^2 / (2 * barrier_width^2))
        return (x - n/2)**2 + barrier_height * np.exp(-(x - n/2)**2 / (2 * (barrier_width * n)**2))
    
    def bitstring_double_well(x: float, params: Dict[str, Any]) -> float:
        """Double-well potential in bitstring basis.
        
        Args:
            x: Input value (bitstring)
            params: Function parameters including:
                - n: System size
                - barrier_height: Height of the barrier between wells
                - barrier_width: Width of the barrier
                
        Returns:
            Potential value
        """
        n = params['n']
        barrier_height = params.get('barrier_height', 1.0)
        barrier_width = params.get('barrier_width', 0.2)
        
        # Convert bitstring to Hamming weight
        hw = int(x * n)
        return hw_double_well(hw, params)
    
    def hw_ham_double_well(t: float, n: int, params: Dict[str, Any]) -> Any:
        """Double-well Hamiltonian in Hamming weight basis.
        
        Args:
            t: Time parameter
            n: System size
            params: Function parameters
            
        Returns:
            Hamiltonian operator as qutip.Qobj
        """
        # Create basis states
        basis = [qt.basis(n+1, i) for i in range(n+1)]
        
        # Create potential term
        V = np.zeros((n+1, n+1))
        for i in range(n+1):
            V[i,i] = hw_double_well(i, params)
        
        # Create kinetic term (hopping between adjacent states)
        K = np.zeros((n+1, n+1))
        for i in range(n):
            K[i,i+1] = np.sqrt(i+1) * np.sqrt(n-i)
            K[i+1,i] = np.sqrt(i+1) * np.sqrt(n-i)
        
        # Combine terms with annealing schedule
        H = (1-t) * K + t * V
        
        return qt.Qobj(H)
    
    def bitstring_ham_double_well(t: float, n: int, params: Dict[str, Any]) -> Any:
        """Double-well Hamiltonian in bitstring basis.
        
        Args:
            t: Time parameter
            n: System size
            params: Function parameters
            
        Returns:
            Hamiltonian operator as qutip.Qobj
        """
        # Create basis states
        basis = [qt.basis(2**n, i) for i in range(2**n)]
        
        # Create potential term
        V = np.zeros((2**n, 2**n))
        for i in range(2**n):
            hw = bin(i).count('1')
            V[i,i] = hw_double_well(hw, params)
        
        # Create kinetic term (single-qubit flips)
        K = np.zeros((2**n, 2**n))
        for i in range(2**n):
            for j in range(n):
                # Flip j-th bit
                neighbor = i ^ (1 << j)
                K[i,neighbor] = 1
        
        # Combine terms with annealing schedule
        H = (1-t) * K + t * V
        
        return qt.Qobj(H)
    
    return CustomFunction(
        name="double_well",
        hw_func=hw_double_well,
        bitstring_func=bitstring_double_well,
        hw_hamiltonian=hw_ham_double_well,
        bitstring_hamiltonian=bitstring_ham_double_well,
        parameters={
            'barrier_height': 1.0,
            'barrier_width': 0.2
        }
    )

# Dictionary to store all custom functions
CUSTOM_FUNCTIONS: Dict[str, CustomFunction] = {
    "example": example_custom_function(),
    "double_well": double_well_function()
}

def register_custom_function(func: CustomFunction) -> None:
    """Register a new custom function.
    
    Args:
        func: CustomFunction instance to register
    """
    CUSTOM_FUNCTIONS[func.name] = func

def get_custom_function(name: str) -> CustomFunction:
    """Get a registered custom function by name.
    
    Args:
        name: Name of the function
        
    Returns:
        CustomFunction instance
        
    Raises:
        KeyError: If function is not found
    """
    if name not in CUSTOM_FUNCTIONS:
        raise KeyError(f"Custom function '{name}' not found")
    return CUSTOM_FUNCTIONS[name] 