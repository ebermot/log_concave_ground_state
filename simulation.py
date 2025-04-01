from main import CustomConvexFunction, run_simulation
from helpers import (
    plot_energy, plot_ground_state, 
    ground_state_animation, plot_gap, plot_min_gap
)
import numpy as np
import argparse

def my_convex_function(x, params):
    """Custom convex function for optimization.
    
    Args:
        x: Input state (integer for Hamming weight basis, list for bitstring basis)
        params: Dictionary containing 'n' and other parameters
    """
    n = params['n']
    alpha = params.get('alpha', 1.0)
    
    if isinstance(x, list):  # bitstring basis
        hw = sum(x)
        return (hw - n/4)**2 * alpha
    else:  # Hamming weight basis
        return (x - n/4)**2 * alpha



if __name__ == "__main__":
    # This allows the script to still be run from command line if needed
    parser = argparse.ArgumentParser(description='Quantum annealing simulation')
    parser.add_argument('n_min', type=int, help='Minimum system size')
    parser.add_argument('n_max', type=int, help='Maximum system size')
    parser.add_argument('--use_hw_basis', action='store_true', help="Use Hamming weight basis")
    parser.add_argument('--num_points', type=int, default=500, help = "number of time points")
    # Add other arguments as needed
    args = parser.parse_args()
    custom_func = CustomConvexFunction(
        func=my_convex_function,
        params={'alpha': 2.0},
        name='quadratic'
        )
    results = run_simulation(args.n_min, args.n_max, custom_func=custom_func, use_hw_basis=args.use_hw_basis, 
    num_points=args.num_points,     # number of time points
    save_results=True,  # save results to file
    sim_params={'gamma': 1.0}  # additional simulation parameter
    )
    for idx, n in enumerate(range(args.n_min, args.n_max+1)):
        plot_energy(
            energies=results['energies'][idx],  # last system size
        n=n,
        levels_to_show=np.arange(2), 
        problem=custom_func.name,
        title_params={'gamma': 1.0}
    )


    # Plot final ground state
    plot_ground_state(
        amp=results['ground_vectors'][idx][-1],  # last time point of last system
        n=n,
        problem=custom_func.name,
        prob=True,  # plot probabilities instead of amplitudes
        dir_path='./plots/',
        sim_params={'gamma': 1.0}
    )

    # Create and display ground state evolution animation
    anim = ground_state_animation(
        ground_vectors= results['ground_vectors'][idx], 
        n=n, 
        problem=custom_func.name,
        output_dir="./plots/ground_state", 
        prob = True, 
        sim_params={'gamma': 1.0}                     
    )
