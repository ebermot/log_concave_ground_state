import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import glob
import re
from scipy.special import binom
from itertools import product
from typing import List, Dict, Union, Tuple, Optional, Callable, Any
from matplotlib import cm, ticker
import os
import pandas as pd

# --- Pauli operators ---

def sigx(Nv: int) -> List[qt.Qobj]:
    """Return all sigma_i^x operators on the Hilbert space.
    
    Args:
        Nv: Number of qubits
        
    Returns:
        List of tensor products of Pauli X operators
    """
    return [qt.tensor([qt.sigmax() if i == j else qt.qeye(2) for i in range(Nv)]) for j in range(Nv)]

def nall(Nv: int) -> List[qt.Qobj]:
    """Generate the set of all n_i operators.
    
    Args:
        Nv: Number of qubits
        
    Returns:
        List of tensor products of number operators
    """
    return [qt.tensor([qt.sigmap() * qt.sigmam() if i == j else qt.qeye(2) for i in range(Nv)]) for j in range(Nv)]


# --- Mathematical helper functions ---


def hamming_weight(x: Union[np.ndarray, List[int]]) -> int:
    """Compute the Hamming weight (number of nonzero elements).
    
    Args:
        x: Input array or list
        
    Returns:
        Number of nonzero elements
    """
    x = np.asarray(x)
    return np.count_nonzero(x)


# --- Hamming weight basis functions ---

def potential_hw(k: int, args: Dict) -> float:
    """Basis transformation of the potential function on the Hamming weight basis.
    
    Args:
        k: Hamming weight
        args: Dictionary containing 'n' and 'potential' parameters
        
    Returns:
        Transformed potential value
    """
    n=args['n']
    bitstrings=[[int(bit) for bit in bitstrings] for bitstrings in product('01', repeat=n)]
    return 1/binom(n, k)**sum(args['potential'](b) for b in bitstrings if hamming_weight(b)==k)


# --- Visualisation ---

def plot_gap(gap: List[float], n: int, problem: str,
            title_params: Optional[Dict[str, float]] = None,
            output_dir: str = './plots/gap',
            filename_prefix: str = 'gap', log_scale:bool =False) -> None:
    """Plot the spectral gap over time.
    
    Args:
        gap: List of gap values
        n: System size
        problem: Problem type
        title_params: Optional dictionary of parameters to include in the plot title.
                     Keys should be parameter names and values should be their values.
                     If None, uses default title format.
        output_dir: Directory to save the plot (default: './plots/gap')
        filename_prefix: Prefix for the output filename (default: 'gap')
    """
    plt.figure()
    
    # Generate time points
    times = np.linspace(0, 1, len(gap))
    
    # Plot the gap
    plt.plot(times, gap)
    plt.xlabel('Time')
    plt.ylabel(r'$\Delta(t) = E_1(t) - E_0(t)$')
    
    # Generate title based on parameters
    if title_params is None:
        plt.title(f'Spectral gap for n={n}')
    else:
        param_str = ', '.join(f'{k}={v}' for k, v in title_params.items())
        plt.title(f'Spectral gap for n={n}, {param_str}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    filename = f'{filename_prefix}_n_{n}'
    if title_params:
        filename += '_' + '_'.join(f'{k}_{v}' for k, v in title_params.items())
    filename += f'_problem_{problem}.pdf'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def plot_energy(energies: Union[np.ndarray, List], n: int, problem: str, 
                labels: Optional[List[str]] = None,
                title_params: Optional[Dict[str, float]] = None,
                output_dir: str = './plots/energy',
                filename_prefix: str = 'energy_spectrum',
                levels_to_show: Optional[List[int]] = None) -> None:
    """Plot multiple energy levels over time.
    
    Args:
        energies: Array or list of shape (n_levels, n_times) containing energy levels
        n: System size
        problem: Problem type
        labels: Optional list of labels for each energy level. If None, 
               generates labels "Level i" for each level.
        title_params: Optional dictionary of parameters to include in the plot title.
                     Keys should be parameter names and values should be their values.
                     If None, uses default title format.
        output_dir: Directory to save the plot (default: './plots/energy')
        filename_prefix: Prefix for the output filename (default: 'energy_spectrum')
        levels_to_show: Optional list of indices of energy levels to display.
                       If None, shows all levels. If specified, only shows
                       the levels at these indices.
    """
    # Convert energies to numpy array if it's a list
    energies = np.asarray(energies)
    
    plt.figure()
  
    # Determine which levels to show
    if levels_to_show is None:
        levels_to_show = list(range(energies.shape[0]))
    
    # Generate default labels if none provided
    if labels is None:
        labels = [f"Level {i}" for i in levels_to_show]
    else:
        labels = [labels[i] for i in levels_to_show]
    
    # Plot each energy level
    for i, (level_idx, label) in enumerate(zip(levels_to_show, labels)):
        # Generate time points
        times = np.linspace(0, 1, len(energies[:, level_idx]))

        plt.plot(times, energies[:, level_idx], label=label)
    
    plt.xlabel('Time')
    plt.ylabel('Energy')
    
    # Generate title based on parameters
    if title_params is None:
        plt.title(f'Energy spectrum for n={n}')
    else:
        param_str = ', '.join(f'{k}={v}' for k, v in title_params.items())
        plt.title(f'Energy spectrum for n={n}, {param_str}')
    
    plt.legend(loc='best')  # Let matplotlib choose the best location
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    filename = f'{filename_prefix}_n_{n}'
    if title_params:
        filename += '_' + '_'.join(f'{k}_{v}' for k, v in title_params.items())
    filename += f'_problem_{problem}.pdf'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_potential(potential: np.ndarray, n: int, problem: str, gamma: float = 1,
                  title_params: Optional[Dict[str, float]] = None,
                  output_dir: str = './plots/potential',
                  filename_prefix: str = 'potential') -> None:
    """Plot the potential function.
    
    Args:
        potential: Potential values (1D array or 2D array)
        n: System size
        problem: Problem type
        gamma: Ratio parameter for density_is problem
        title_params: Optional dictionary of parameters to include in the plot title
        output_dir: Directory to save the plot (default: './plots/potential')
        filename_prefix: Prefix for the output filename (default: 'potential')
    """
    fig, ax = plt.subplots()
    
    if isinstance(potential, list):
        potential = np.array(potential)
    
    if len(potential.shape) == 1:
        # 1D potential
        x_value = np.arange(0, n+1)
        plt.plot(x_value, potential)
        plt.xlabel(r'$|x|$')
        plt.ylabel(r'$V(|x|)$')
        title = r'Potential $V(|x|)$'
        
    elif len(potential.shape) == 2:
        # 2D potential
        x_value = np.linspace(0, potential.shape[0], potential.shape[0])
        y_value = np.linspace(0, potential.shape[1], potential.shape[1])
        X, Y = np.meshgrid(x_value, y_value)
        cs = ax.contourf(X, Y, potential, levels=100, cmap=cm.plasma)
        cbar = fig.colorbar(cs)
        plt.xlabel(r'$|x|$')
        plt.ylabel(r'$|y|$')
        title = r'Potential $V(|x|, |y|)$'
    
    # Generate title based on parameters
    if title_params is None:
        plt.title(f'{title} for n={n}')
    else:
        param_str = ', '.join(f'{k}={v}' for k, v in title_params.items())
        plt.title(f'{title} for n={n}, {param_str}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    filename = f'{filename_prefix}_n_{n}'
    if title_params:
        filename += '_' + '_'.join(f'{k}_{v}' for k, v in title_params.items())
    filename += f'_problem_{problem}.pdf'
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()
def generate_filename(n: int, problem: str, frame_idx: Optional[int] = None, 
                     for_animation: bool = False) -> str:
    """Generate consistent filename for ground state plots and animations.
    
    Args:
        n: System size
        problem: Problem type
        frame_idx: Optional frame index for animation frames
        for_animation: If True, generate PNG filename, else PDF
    """
    if frame_idx is not None:
        # Frame for animation
        return f'frame_{frame_idx:03d}.png'
    else:
        # Single plot or final animation
        ext = 'png' if for_animation else 'pdf'
        return f'n_{n}_problem_{problem}.{ext}'

def plot_ground_state(amp: np.ndarray, n: int, problem: str, 
                     dir_path: Optional[str] = None, 
                     prob: bool = True, 
                     sim_params: dict = None,
                     for_animation: bool = False,
                     frame_idx: Optional[int] = None,
                     show_plot: bool = True) -> Optional[str]:
    """Plot the ground state.
    
    Returns:
        Optional[str]: Path to saved file if dir_path is provided
    """
    if prob:
        amp = np.abs(amp) ** 2
    
    title = rf'Ground state for n={n}'
    if sim_params:
        param_str = ', '.join([f'{k}={v}' for k, v in sim_params.items()])
        title += f'\n({param_str})'
    
    plt.figure()
    if len(amp.shape) > 1:
        im = plt.imshow(amp, cmap='viridis', interpolation='nearest', origin="lower")
        cbar = plt.colorbar(im)
        cbar.set_label('Probability' if prob else 'Amplitude')
        plt.title(title)
    else:
        plt.bar(np.arange(0, len(amp)), amp)
        plt.title(title)
        plt.xlabel(r'$|x|$')
        plt.ylabel('Probability' if prob else 'Amplitude')
    
    saved_file = None
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        filename = generate_filename(n, problem, frame_idx, for_animation)
        saved_file = os.path.join(dir_path, filename)
        plt.savefig(saved_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return saved_file

def ground_state_figure(ground_vectors: List[np.ndarray], n: int, problem: str, 
                       temp_dir: str, prob: bool = True, 
                       sim_params: dict = None) -> List[str]:
    """Generate figures for ground state evolution."""
    os.makedirs(temp_dir, exist_ok=True)
    times = np.linspace(0, 1, len(ground_vectors))
    generated_files = []
    
    for k, state in enumerate(ground_vectors):
        current_params = sim_params.copy() if sim_params else {}
        current_params['t'] = f'{times[k]:.2f}'
        
        saved_file = plot_ground_state(
            amp=state,
            n=n,
            problem=problem,
            dir_path=temp_dir,
            prob=prob,
            sim_params=current_params,
            for_animation=True,
            frame_idx=k,
            show_plot=False
        )
        
        if saved_file:
            generated_files.append(saved_file)
            
    return generated_files

def ground_state_animation(ground_vectors: List[np.ndarray], n: int, problem: str,
                         output_dir: str, prob: bool = True, 
                         sim_params: dict = None,
                         interval: int = 500, 
                         repeat_delay: int = 10,
                         cleanup: bool = True) -> animation.Animation:
    """Create animation of ground state evolution."""
    temp_dir = os.path.join(output_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Generating frames...")
    png_files = ground_state_figure(
        ground_vectors=ground_vectors,
        n=n,
        problem=problem,
        temp_dir=temp_dir,
        prob=prob,
        sim_params=sim_params
    )
    
    print(f"Created {len(png_files)} frames")
    print("Creating animation...")
    
    image_array = [Image.open(file) for file in png_files]
    fig, ax = plt.subplots()
    im = ax.imshow(image_array[0], animated=True)
    
    def update(i: int) -> Tuple[Any]:  # Changed return type hint
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(image_array),
        interval=interval, blit=True, repeat_delay=repeat_delay
    )

    # Save animation using consistent filename
    animation_file = os.path.join(output_dir, 
                                generate_filename(n, problem, for_animation=True))
    anim.save(animation_file)
    
    if cleanup:
        print("Cleaning up temporary files...")
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    
    return anim

def plot_min_gap(results_file: Union[str, dict], nlist: List[int], problem: str,
                title_params: Optional[Dict[str, float]] = None,
                output_dir: str = './plots/min_gap',
                filename_prefix: str = 'min_gap',
                log_scale: bool = False,
                log_log: bool = False,
                n_range: Optional[Tuple[int, int]] = None) -> None:
    """Plot the minimum gap as a function of system size.
    
    Args:
        results_file: Path to the results file (HDF5 format) or dictionnary
        nlist: List of system sizes
        problem: Problem type
        title_params: Optional dictionary of parameters to include in the plot title
        output_dir: Directory to save the plot (default: './plots/min_gap')
        filename_prefix: Prefix for the output filename (default: 'min_gap')
        log_scale: If True, use log scale on y-axis
        log_log: If True, use log scale on both x and y axes
        n_range: Optional tuple (min_n, max_n) to filter system sizes
    """
    # Handle input type
    if isinstance(results_file, str):
        results = pd.read_hdf(results_file, key='hamil')
    elif isinstance(results_file, dict):
        results = results_file
    else:
        raise TypeError("results_file must be either a string path or a dictionnary")
    
    
    # Calculate minimum gaps
    min_gaps = []
    filtered_nlist = []
    for idx, n in enumerate(nlist):
        if n_range is None or (n_range[0] <= n <= n_range[1]):
            gap = results['gap'][idx]
            min_gaps.append(min(gap))
            filtered_nlist.append(n)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_nlist, min_gaps)
    
    # Add labels and grid
    plt.xlabel(r'System size, $N$')
    plt.ylabel(r'$\Delta_{min}$')
    plt.grid(True)
    
    # Set log scales if requested
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
    elif log_scale:
        plt.yscale('log')
    
    # Generate title based on parameters
    if title_params is None:
        plt.title(f'Minimum gap vs system size for {problem}')
    else:
        param_str = ', '.join(f'{k}={v}' for k, v in title_params.items())
        plt.title(f'Minimum gap vs system size for {problem}, {param_str}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    filename = f'{filename_prefix}_{problem}'
    if title_params:
        filename += '_' + '_'.join(f'{k}_{v}' for k, v in title_params.items())
    if log_log:
        filename += '_loglog'
    elif log_scale:
        filename += '_logy'
    if n_range:
        filename += f'_n{n_range[0]}-{n_range[1]}'
    filename += '.pdf'
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.show()

def create_perturbed_function(base_func: Callable, perturbation_func: Callable) -> Callable:
    """Create a new function that combines a base function with a perturbation.
    
    Args:
        base_func: The base function to perturb
        perturbation_func: The perturbation function that takes (x, args) and returns a value
        
    Returns:
        A new function that returns base_func(x) + perturbation_func(x) if condition is met,
        otherwise returns base_func(x)
    """
    def perturbed_func(x: int, args: Dict) -> int:
        return base_func(x, args) + perturbation_func(x, args)
    return perturbed_func

def create_perturbed_bitstring_function(base_func: Callable, perturbation_func: Callable) -> Callable:
    """Create a new bitstring function that combines a base function with a perturbation.
    
    Args:
        base_func: The base bitstring function to perturb
        perturbation_func: The perturbation function that takes (x, args) and returns a value
        
    Returns:
        A new function that applies the perturbation to the Hamming weight
    """
    def perturbed_bitstring_func(x: List[int], args: Dict) -> int:
        hw = hamming_weight(x)
        return base_func(x, args) + perturbation_func(hw, args)
    return perturbed_bitstring_func