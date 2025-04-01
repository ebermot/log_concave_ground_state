# Log-Concave Ground State  

## About  

This project explores the properties of ground states of an annealing Hamiltonian with convex potentials.  

## Installation  

To run the project, you need to install the following dependencies:  

- `numpy`  
- `pandas`  
- `qutip`  
- `tables`  
- `scipy`  
- `matplotlib`  

You can install them directly using:  

```sh
pip install -r requirements.txt
```

## Code Structure  

- **`hamiltonian.py`** – Defines the basic structure for Hamiltonians.  
- **`helpers.py`** – Contains all plotting functions.  
- **`main.py`** – Includes all core routines (e.g., eigenstate computation, overlap calculations with the final ground state and first excited state).  

## Usage  

You can use the following Python example to run a simulation:  

```python
from main import CustomConvexFunction, run_simulation
from helpers import (
    plot_energy, plot_ground_state,
    ground_state_animation, plot_gap, plot_min_gap
)
from IPython.display import HTML
import numpy as np

np.set_printoptions(legacy='1.25')

def my_convex_function(x, params):
    """Custom convex function for optimization.
    
    Args:
        x: Input state (integer for Hamming weight basis, list for bitstring basis).
        params: Dictionary containing 'n' and other parameters.
    
    Returns:
        Float value of the function applied to `x`.
    """
    # YOUR CODE HERE
    return 0

# Create function object with parameters
custom_func = CustomConvexFunction(
    func=my_convex_function,
    params={'alpha': 2.0},
    name='quadratic'
)

# Run quantum annealing simulation
results = run_simulation(
    n_min=5,
    n_max=5,
    custom_func=custom_func,
    use_hw_basis=True,  # Use Hamming weight basis
    num_points=500,     # Number of time points
    save_results=True,  # Save results to file
    sim_params={'gamma': 1.0}  # Additional simulation parameters
)
```

### **Plotting Functions**  

Once you have the simulation results, you can visualize them using the following functions:  

- **`plot_energy`** – Plots energy levels (you can specify how many levels to display).  
- **`plot_min_gap`** – Plots the minimal energy gap (depends on the number of data points used).  
- **`plot_ground_state`** – Plots the ground state probability (or amplitude, if preferred).  
- **`ground_state_animation`** – Creates an animation of the ground state probability in the Hamming weight (HW) basis (or amplitude, if preferred) as time progresses from 0 to 1.  

### **Running from the Terminal**  

You can also run a simulation directly from the command line:  

```sh
python simulation.py 5 5 --use_hw_basis
```

