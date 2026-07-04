import time
import torch
import numpy as np

# Force CPU execution for clear sequential execution timing
device = torch.device("cpu")

# =====================================================================
# 1. DEFINE THE MODEL OPERATOR
# =====================================================================
class SparseOptimizedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 4096 x 4096 weight matrix simulating a hidden layer
        self.weight = torch.nn.Parameter(torch.randn(4096, 4096, device=device))

    def forward(self, x):
        """
        Simulating a naive DL compiler optimization.
        If a block of weights is entirely zero, the compiler introduces a data-dependent
        branch to skip the expensive dot-product computation to save power/clock cycles.
        """
        # --- The Compiler Lowering Invariant Loop (Simulated) ---
        if torch.all(self.weight == 0):
            # Optimizing pass: skip math, return zeros
            return torch.zeros(x.shape[0], 4096, device=device)
        else:
            # Standard matrix multiplication
            return torch.matmul(x, self.weight)

# Initialize our baseline model
model = SparseOptimizedLinear()
public_input = torch.randn(1, 4096, device=device)

# =====================================================================
# 2. BASELINE PHASE (Before Compilation / Interpreter Mode)
# =====================================================================
print("--- PHASE 1: Running Eager Python Interpreter Mode ---")

# Scenario Alpha: Secret weights are completely random values
torch.nn.init.normal_(model.weight)
t0 = time.perf_counter()
_ = model(public_input)
time_alpha_eager = time.perf_counter() - t0

# Scenario Beta: Secret weights are entirely zeros
torch.nn.init.zeros_(model.weight)
t0 = time.perf_counter()
_ = model(public_input)
time_beta_eager = time.perf_counter() - t0

print(f"Eager Time (Alpha - Random Weights): {time_alpha_eager:.6f} seconds")
print(f"Eager Time (Beta - Zero Weights):   {time_beta_eager:.6f} seconds")
print(f"Eager Delta (Leakage Signature):     {abs(time_alpha_eager - time_beta_eager):.6f} seconds")
# (In eager Python, both branches hit the evaluation engine identically, delta is close to 0)

# =====================================================================
# 3. LEAKAGE PHASE (After Graph Compilation / Induction)
# =====================================================================
print("\n--- PHASE 2: Compiling the Graph to Machine Execution ---")

# Compile the model graph using PyTorch's Inductor backend
compiled_model = torch.compile(model, fullgraph=True)

# Warmup pass (compilers require a warmup run to bake the machine kernel)
torch.nn.init.normal_(model.weight)
_ = compiled_model(public_input)

# Test Scenario Alpha (Random Weights) on Compiled Graph
torch.nn.init.normal_(model.weight)
timings_alpha = []
for _ in range(50): # Run multiple iterations to filter out OS noise
    t0 = time.perf_counter()
    _ = compiled_model(public_input)
    timings_alpha.append(time.perf_counter() - t0)
mean_alpha_compiled = np.mean(timings_alpha)

# Test Scenario Beta (Zero Weights) on Compiled Graph
torch.nn.init.zeros_(model.weight)
timings_beta = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = compiled_model(public_input)
    timings_beta.append(time.perf_counter() - t0)
mean_beta_compiled = np.mean(timings_beta)

print(f"Compiled Time (Alpha - Random Weights): {mean_alpha_compiled:.6f} seconds")
print(f"Compiled Time (Beta - Zero Weights):   {mean_beta_compiled:.6f} seconds")

# Calculate execution time discrepancy
delta_compiled = abs(mean_alpha_compiled - mean_beta_compiled)
ratio = (mean_alpha_compiled / mean_beta_compiled) if mean_beta_compiled > 0 else 0

print(f"Compiled Delta (Leakage Signature):     {delta_compiled:.6f} seconds")
print(f"Execution Speed Variance:               {ratio:.2f}x faster execution!")

if delta_compiled > (abs(time_alpha_eager - time_beta_eager) * 2):
    print("\n[CRITICAL SECURITY ALERT]: CONFIDENTIALITY INVARIANT BROKEN.")
    print("The compiler pass introduced a data-dependent execution signature.")
    print("An external attacker can infer the entropy of secret weights via public API latency.")
else:
    print("\nNon-interference invariant maintained.")
