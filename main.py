"""Quantum annealing simulation for various physical systems including custom convex functions."""

from typing import Dict, List, Optional, Tuple, Callable
import argparse
import numpy as np
import pandas as pd
import qutip as qt
from itertools import product
from helpers import *
from hamiltonians import *

class CustomConvexFunction:
    """Class to handle custom convex functions for quantum annealing."""
    def __init__(self, func: Callable, params: Dict = None, name: str = "custom"):
        """
        Args:
            func: The convex function to use
            params: Dictionary of parameters for the function
            name: Name identifier for the function
        """
        self.func = func
        self.params = params or {}
        self.name = name

def initialize_results() -> Dict:
    """Initialize the results dictionary."""
    return {
        "ground_state_energy": [],
        "first_excited_energy": [],
        "energies": [],
        "bitstrings": [],
        "ground_vectors": [],
        "gap": [],
        "potential": [],
        "G0": [],
        "G1": [],
        "parameters": {},  # Store simulation parameters
        "times": None      # Store time points
    }

def compute_hamiltonians(n: int, times: np.ndarray, custom_func: Optional[CustomConvexFunction] = None,
                        use_hw_basis: bool = True, **kwargs) -> Tuple[List[qt.Qobj], List[float]]:
    """Compute Hamiltonians and potential for given parameters.
    
    Args:
        n: System size
        times: Time points
        custom_func: Optional custom convex function
        use_hw_basis: If True, use Hamming weight basis, otherwise use bitstring basis
        **kwargs: Additional parameters for Hamiltonian construction
    
    Returns:
        Tuple of (hamiltonian_list, potential_list)
    """
    if custom_func is not None:
        # Update parameters with n and any additional kwargs
        params = custom_func.params.copy()
        params.update({'n': n, **kwargs})
        
        # Choose appropriate annealing schedule based on basis
        if use_hw_basis:
            annealing_schedule = hs_hw_annealing
        else:
            annealing_schedule = hs_annealing
            
        # Create Hamiltonians for each time point
        ham_list = [annealing_schedule(t, custom_func.func, params) for t in times]
        
        # Compute potential values (for visualization)
        if use_hw_basis:
            x_value = np.arange(0, n+1)
            potential = [custom_func.func(x, params) for x in x_value]
        else:
            basis = [[int(bit) for bit in bitstring] for bitstring in product('01', repeat=n)]
            potential = [custom_func.func(b, params) for b in basis]
        
        return ham_list, potential
    else:
        raise ValueError("No function provided for Hamiltonian construction")

def compute_eigenstates(ham: qt.Qobj, final_gs_states: List[qt.Qobj], is_final: bool = False) -> Tuple:
    """Compute eigenstates and related quantities for a Hamiltonian.
    
    Args:
        ham: Hamiltonian to diagonalize
        final_gs_states: List of final ground states
        is_final: Whether this is the final time step
        
    Returns:
        Tuple of computed quantities
    """
    values, states = ham.eigenstates(sparse=True, sort='low')
    
    if not is_final:
        gs_energy = values[0]
        fs_energy = values[1]
        gap = fs_energy - gs_energy
        state_vector = states[0].data.toarray().reshape(states[0].shape[0])
        
        # Compute overlaps
        overlaps_G0 = sum(np.abs(final_gs.overlap(states[0]))**2 for final_gs in final_gs_states)
        overlaps_G1 = sum(np.abs(final_gs.overlap(states[1]))**2 for final_gs in final_gs_states)
        
        return gs_energy, fs_energy, gap, state_vector, values, overlaps_G0, overlaps_G1
    
    else:
        gs_value = min(values)
        fs_value = np.unique(values)[1]
        
        # Collect ground and first excited states
        gs_vectors = [states[i] for i in range(len(values)) if np.isclose(values[i], gs_value)]
        fs_vectors = [states[i] for i in range(len(values)) if np.isclose(values[i], fs_value)]
        
        gap = 0 if len(gs_vectors) > 1 else fs_value - gs_value
        uniform_gs = sum(gs_vectors).unit()
        state_vector = uniform_gs.data.toarray().reshape(states[0].shape[0])
        
        # Compute overlaps
        overlaps_G0 = sum(np.abs(final_gs.overlap(vec))**2 for final_gs in final_gs_states for vec in gs_vectors)
        overlaps_G1 = sum(np.abs(final_gs.overlap(vec))**2 for final_gs in final_gs_states for vec in fs_vectors)
        
        return gs_value, fs_value, gap, state_vector, values, overlaps_G0, overlaps_G1

def process_time_evolution(ham_list: List[qt.Qobj], times: np.ndarray, 
                         final_gs_states: List[qt.Qobj]) -> Dict:
    """Process time evolution for a given Hamiltonian list.
    
    Args:
        ham_list: List of Hamiltonians at different times
        times: Time points
        final_gs_states: Final ground states
        
    Returns:
        Dictionary containing computed quantities
    """
    temp_gs, temp_fs, temp_gap = [], [], []
    temp_state, temp_G0, temp_G1 = [], [], []
    temp_energies = []
    
    for idx, t in enumerate(times):
        is_final = idx == len(times) - 1
        gs_energy, fs_energy, gap, state_vector, values, G0, G1 = compute_eigenstates(
            ham_list[idx], final_gs_states, is_final
        )
        
        temp_gs.append(gs_energy)
        temp_fs.append(fs_energy)
        temp_gap.append(gap)
        temp_state.append(state_vector)
        temp_energies.append(values)
        temp_G0.append(G0)
        temp_G1.append(G1)
    
    return {
        'gs': temp_gs,
        'fs': temp_fs,
        'gap': temp_gap,
        'state': temp_state,
        'G0': temp_G0,
        'G1': temp_G1,
        'energies': temp_energies
    }

def run_simulation(n_min: int, n_max: int, custom_func: Optional[CustomConvexFunction] = None,
                  num_points: int = 500, save_results: bool = False, 
                  output_dir: str = './results', **kwargs) -> Dict:
    """Run quantum annealing simulation.
    
    Args:
        n_min: Minimum system size
        n_max: Maximum system size
        custom_func: Optional custom convex function
        num_points: Number of time points
        save_results: Whether to save results to file
        output_dir: Directory for saving results
        **kwargs: Additional parameters for the simulation
    
    Returns:
        Dictionary containing simulation results
    """
    results = initialize_results()
    times = np.linspace(0, 1, num_points)
    results['times']=[]
    results['parameters']=[]
    params={'n_min': n_min, 'n_max': n_max, **kwargs}
    if custom_func is not None:
        params['function_name'] = custom_func.name
        params.update(custom_func.params)

    for n in range(n_min, n_max + 1):
        print(f'Processing N={n}')
        
        # Compute Hamiltonians and potentials
        ham_list, potential_list = compute_hamiltonians(n, times, custom_func, **kwargs)
        
        # Get final ground states
        final_values, final_states = ham_list[-1].eigenstates(sparse=True, sort='low')
        final_gs_value = min(final_values)
        final_gs_states = [final_states[i] for i in range(len(final_values)) 
                          if np.isclose(final_values[i], final_gs_value)]
        
        # Process time evolution
        temp_results = process_time_evolution(ham_list, times, final_gs_states)
        
        # Store results
        results['ground_state_energy'].append(temp_results['gs'])
        results['first_excited_energy'].append(temp_results['fs'])
        results['energies'].append(temp_results['energies'])
        results['gap'].append(temp_results['gap'])
        results['bitstrings'].append(final_states)
        results['ground_vectors'].append(temp_results['state'])
        results['G0'].append(temp_results['G0'])
        results['G1'].append(temp_results['G1'])
        results['potential'].append(potential_list)
        results['times'].append(times)
        results['parameters'].append(params)
            # Diagnostic cell to check all array lengths in results
    if save_results:
        save_to_file(results, output_dir, custom_func)
        
    return results

def save_to_file(results: Dict, output_dir: str, custom_func: CustomConvexFunction):
    """Save results to HDF5 file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from parameters
    params = results['parameters'][0]
    base_name = f"{custom_func.name}_n{params['n_min']}-{params['n_max']}"
    
    # Create parameter string by joining key-value pairs with underscores
    param_parts = []
    for k, v in params.items():
        if k not in ['n_min', 'n_max', 'function_name']:
            if isinstance(v, dict):
                # Handle nested dictionary
                for sub_k, sub_v in v.items():
                    # Format the nested value
                    if isinstance(sub_v, float):
                        v_str = f"{sub_v:.2f}".rstrip('0').rstrip('.')
                    else:
                        v_str = str(sub_v)
                    param_parts.append(f"{sub_k}_{v_str}")
            else:
                # Handle non-dictionary values
                if isinstance(v, float):
                    v_str = f"{v:.2f}".rstrip('0').rstrip('.')
                else:
                    v_str = str(v)
                param_parts.append(f"{k}_{v_str}")
    
    # Join all parameter parts with underscores
    param_str = '_'.join(param_parts)
    
    filename = f"{base_name}_{param_str}.h5"
    
    # Save to HDF5
    df = pd.DataFrame(results)
    df.to_hdf(os.path.join(output_dir, filename), key='hamil')

    
if __name__ == "__main__":
    # This allows the script to still be run from command line if needed
    parser = argparse.ArgumentParser(description='Quantum annealing simulation')
    parser.add_argument('n_min', type=int, help='Minimum system size')
    parser.add_argument('n_max', type=int, help='Maximum system size')
    # Add other arguments as needed
    args = parser.parse_args()
    
    results = run_simulation(args.n_min, args.n_max)