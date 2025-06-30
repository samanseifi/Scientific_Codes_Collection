import numpy as np
import matplotlib.pyplot as plt
import os
import numba # Numba can still be useful for the non-linear terms in real space
import time # For timing the simulation

# --- Configuration Parameters ---
class SimConfig:
    def __init__(self):
        # Grid dimensions
        self.Nx, self.Ny = 256, 256
        self.dx, self.dy = 2.0, 2.0
        self.Lx, self.Ly = self.Nx * self.dx, self.Ny * self.dy

        # Simulation steps and saving
        self.total_steps = 5000
        self.save_interval = 10 # Increased save interval for longer runs
        self.output_dir = "simulation_snapshots_fft" # New directory for FFT results

        # Model parameters from the paper
        self.num_orientations = 30
        self.C_alpha = 0.05 # Solubility in the matrix phase
        self.C_beta = 0.95  # Solubility in the second phase
        self.C_m = (self.C_alpha + self.C_beta) / 2 # Mid-point concentration

        # Initial average concentration for Ostwald ripening
        # Adjusted for desired volume fraction, e.g., ~25% for C_beta phase
        # C_avg = (1 - V_f) * C_alpha + V_f * C_beta
        # For V_f = 0.25: C_avg = 0.75 * 0.05 + 0.25 * 0.95 = 0.0375 + 0.2375 = 0.275
        self.initial_average_concentration = 0.275 # Aiming for ~25% volume fraction of C_beta

        self.A = 2.0
        self.B = self.A / ((self.C_beta - self.C_m)**2)
        self.gamma_f2 = 2.0 / ((self.C_beta - self.C_alpha)**2)
        self.delta_f2 = 1.0
        self.D_alpha_f1 = self.gamma_f2 / (self.delta_f2**2) # D_alpha in f1(C)
        self.D_beta_f1 = self.gamma_f2 / (self.delta_f2**2)  # D_beta in f1(C)
        self.epsilon_ij = 3.0

        # Gradient energy coefficients
        self.kappa_C = 2.0
        self.kappa_eta = 2.0

        # Kinetic coefficients
        self.D_mobility = 1.0 # M_C in the semi-implicit equations
        self.L_mobility = 1.0 # L_i in the semi-implicit equations

        # Time step for integration - can be larger with semi-implicit
        self.dt = 1.0 # Significantly larger dt now possible, experiment with this!

        # Initial noise for field initialization
        self.initial_noise_amplitude = 0.001

# --- Core Simulation Functions ---

# Numba is still useful for the real-space derivative calculation
@numba.jit(nopython=True, fastmath=True)
def calculate_free_energy_derivatives_numba(C, eta_fields, A, B, C_m, C_alpha, C_beta, D_alpha_f1, D_beta_f1, gamma_f2, delta_f2, epsilon_ij, num_orientations):
    """
    Numba-optimized version of calculate_free_energy_derivatives.
    Calculates only the local (non-gradient) derivatives of the free energy.
    """
    Nx, Ny = C.shape
    dF_dC = np.empty_like(C)
    dF_d_eta_i = np.empty_like(eta_fields)
    sum_eta_j_sq = np.empty_like(C)

    # Calculate sum_eta_j_sq first
    for i in range(Nx):
        for j in range(Ny):
            current_sum = 0.0
            for k in range(num_orientations):
                current_sum += eta_fields[k, i, j]**2
            sum_eta_j_sq[i, j] = current_sum

    # dF_dC calculation
    for i in range(Nx):
        for j in range(Ny):
            # f1(C) derivative
            dF_dC_f1_val = -A * (C[i, j] - C_m) + \
                           B * (C[i, j] - C_m)**3 + \
                           D_alpha_f1 * (C[i, j] - C_alpha)**3 + \
                           D_beta_f1 * (C[i, j] - C_beta)**3

            # f2(C, eta_i) derivative wrt C (summed)
            dF_dC_f2_sum_val = 0.0
            for k in range(num_orientations):
                dF_dC_f2_sum_val += -gamma_f2 * (C[i, j] - C_alpha) * eta_fields[k, i, j]**2
            
            dF_dC[i, j] = dF_dC_f1_val + dF_dC_f2_sum_val

    # dF_d_eta_i calculation
    for k in range(num_orientations):
        for i in range(Nx):
            for j in range(Ny):
                # From f2(C, eta_i)
                dF_d_eta_i_f2_val = -gamma_f2 * (C[i, j] - C_alpha)**2 * eta_fields[k, i, j] + \
                                    delta_f2 * eta_fields[k, i, j]**3

                # From f3(eta_i, eta_j)
                cross_term_val = epsilon_ij * eta_fields[k, i, j] * \
                                 (sum_eta_j_sq[i, j] - eta_fields[k, i, j]**2)

                dF_d_eta_i[k, i, j] = dF_d_eta_i_f2_val + cross_term_val

    return dF_dC, dF_d_eta_i

# K-space grid generation (no Numba here as it's a one-time setup)
def create_k_space_grid(Nx, Ny, dx, dy):
    """Generates the squared wavenumber grid for 2D FFT."""
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2
    return k_squared

def run_simulation_semi_implicit():
    """
    Main simulation loop for Ostwald ripening using Semi-Implicit FFT method.
    """
    config = SimConfig()

    # Create output directory if it doesn't exist
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Created output directory: {config.output_dir}")

    # Initialize fields
    C = config.initial_average_concentration + config.initial_noise_amplitude * (np.random.rand(config.Nx, config.Ny) - 0.5)
    eta_fields = np.zeros((config.num_orientations, config.Nx, config.Ny))
    for i in range(config.num_orientations):
        eta_fields[i] = config.initial_noise_amplitude * (np.random.rand(config.Nx, config.Ny) - 0.5)

    print(f"Simulation started with dt = {config.dt} (Semi-Implicit FFT)")
    print(f"Grid size: {config.Nx}x{config.Ny}, dx={config.dx}")

    # Pass config values to Numba function as individual arguments
    A, B, C_m, C_alpha, C_beta = config.A, config.B, config.C_m, config.C_alpha, config.C_beta
    D_alpha_f1, D_beta_f1 = config.D_alpha_f1, config.D_beta_f1
    gamma_f2, delta_f2 = config.gamma_f2, config.delta_f2
    epsilon_ij, num_orientations = config.epsilon_ij, config.num_orientations
    kappa_C, kappa_eta = config.kappa_C, config.kappa_eta
    D_mobility, L_mobility = config.D_mobility, config.L_mobility
    dx, dy, dt = config.dx, config.dy, config.dt

    # Precompute k-space operators (done once)
    k_squared = create_k_space_grid(config.Nx, config.Ny, dx, dy)
    k_four = k_squared**2 # For the Cahn-Hilliard bi-Laplacian term

    # Add a small epsilon to avoid division by zero or very large numbers at k=0
    # This specifically for the k=0 component in the denominator
    # For k=0, the equations simplify, but numerically, if k_squared is exactly 0, 1/k_squared is problematic
    # For k=0, the dynamics are typically average concentration/order parameter conservation.
    # The denominators 1 + dt * M_C * kappa_C * k_four and 1 + dt * L_i * kappa_i * k_squared will be 1 at k=0
    # so simply dividing by 1 is fine.
    # We use np.divide with 'where' argument to handle potential zeros safely.

    start_time = time.time() # Start timer

    for step in range(config.total_steps):
        # 1. Compute local (non-gradient) derivatives of f0 in real space
        dF_dC_local, dF_d_eta_i_local = calculate_free_energy_derivatives_numba(
            C, eta_fields, A, B, C_m, C_alpha, C_beta, D_alpha_f1, D_beta_f1,
            gamma_f2, delta_f2, epsilon_ij, num_orientations
        )

        # 2. Fourier transform current fields and local derivatives
        hat_C = np.fft.fftn(C)
        hat_dF_dC_local = np.fft.fftn(dF_dC_local)
        
        # Transform each eta_field slice separately, then stack results
        # Or, transform all at once using axes=(1,2) for spatial dimensions
        hat_eta_fields = np.fft.fftn(eta_fields, axes=(1, 2))
        hat_dF_d_eta_i_local = np.fft.fftn(dF_d_eta_i_local, axes=(1, 2))

        # 3. Update C in Fourier space (Semi-Implicit Cahn-Hilliard)
        # Equation: hat_C_new = (hat_C + dt * D_mobility * (-k_squared) * hat_dF_dC_local) / (1 + dt * D_mobility * kappa_C * k_four)
        numerator_C = hat_C + dt * D_mobility * (-k_squared) * hat_dF_dC_local
        denominator_C = 1 + dt * D_mobility * kappa_C * k_four
        
        # Safely divide, avoiding issues where denominator_C might be zero (though unlikely with k_four and positive parameters)
        hat_C_new = np.divide(numerator_C, denominator_C, out=np.zeros_like(numerator_C, dtype=np.complex128), where=denominator_C!=0)
        
        # 4. Update eta_i in Fourier space (Semi-Implicit Ginzburg-Landau)
        # Equation: hat_eta_i_new = (hat_eta_i - dt * L_mobility * hat_dF_d_eta_i_local) / (1 + dt * L_mobility * kappa_eta * k_squared)
        
        numerator_eta = hat_eta_fields - dt * L_mobility * hat_dF_d_eta_i_local
        denominator_eta = 1 + dt * L_mobility * kappa_eta * k_squared
        
        # Apply the update for each eta_i field
        hat_eta_fields_new = np.divide(numerator_eta, denominator_eta, out=np.zeros_like(numerator_eta, dtype=np.complex128), where=denominator_eta!=0)


        # 5. Inverse Fourier transform back to real space
        C = np.fft.ifftn(hat_C_new).real
        eta_fields = np.fft.ifftn(hat_eta_fields_new, axes=(1, 2)).real # Take real part as fields are real

        # Clipping (still useful for robustness, as non-linear terms are explicit)
        eta_fields = np.clip(eta_fields, -1.1, 1.1)
        C = np.clip(C, config.C_alpha - 0.1, config.C_beta + 0.1)

        # Define fixed color scale limits
        C_min = config.C_alpha - 0.1  # e.g., ~ -0.05
        C_max = config.C_beta + 0.1   # e.g., ~ 1.05

        # Save snapshots for visualization
        if (step + 1) % config.save_interval == 0:
            print(f"Step: {step + 1}/{config.total_steps}")

            # --- Concentration Field ---
            plt.figure(figsize=(8, 8))
            plt.imshow(C, cmap='jet', origin='lower',
                    extent=[0, config.Lx, 0, config.Ly],
                    vmin=C_min, vmax=C_max)  # Fixed scale
            plt.colorbar(label=r'Concentration ($C$)')
            plt.title(r'Concentration Field at Step {}'.format(step + 1))
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.savefig(os.path.join(config.output_dir, f'concentration_step_{step + 1:06d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

            # --- Microstructure Field (Sum of ηᵢ²) ---
            combined_eta_sq = np.sum(eta_fields**2, axis=0)
            plt.figure(figsize=(8, 8))
            plt.imshow(combined_eta_sq, cmap='jet', origin='lower',
                    extent=[0, config.Lx, 0, config.Ly],
                    vmin=0, vmax=1.1 * np.max(eta_fields**2))  # Optional fixed range for comparability
            plt.xticks([])  # Remove ticks for clean GIF
            plt.yticks([])
            plt.savefig(os.path.join(config.output_dir, f'microstructure_step_{step + 1:06d}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    return C, eta_fields

# --- Run the simulation ---
if __name__ == "__main__":
    final_C, final_eta = run_simulation_semi_implicit()

    # Display final state (optional, as snapshots are saved)
    plt.figure(figsize=(8, 8))
    plt.imshow(final_C, cmap='viridis', origin='lower',
               extent=[0, final_C.shape[1]*2.0, 0, final_C.shape[0]*2.0])
    plt.colorbar(label='Concentration (C)')
    plt.title(f'Final Concentration Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.sum(final_eta**2, axis=0), cmap='gray', origin='lower',
               extent=[0, final_eta.shape[2]*2.0, 0, final_eta.shape[1]*2.0])
    plt.colorbar(label='Sum of eta_i^2')
    plt.title(f'Final Microstructure (Sum of eta_i^2)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()