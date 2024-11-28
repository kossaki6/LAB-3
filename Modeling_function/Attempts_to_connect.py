import json
import sympy as sp
import numpy as np

# Load the JSON data
with open("equations.json", "r") as f:  
    data = json.load(f)

# Define symbols
x1, x2, t = sp.symbols('x1 x2 t')

# Define expressions for functions
y_s = sp.sympify(data["y(s)"])
G_s = sp.sympify(data["G(s)"])
u_s = sp.sympify(data["u(s)"])

# Define differential map with specific orders
differential_map = {
    "delta_t_1": sp.Derivative(y_s, t),
    "delta_t_2": sp.Derivative(y_s, t, t),
    "delta_x1_1": sp.Derivative(y_s, x1),
    "delta_x1_2": sp.Derivative(y_s, x1, x1),
    "delta_x2_1": sp.Derivative(y_s, x2),
    "delta_x2_2": sp.Derivative(y_s, x2, x2),
    "1": y_s  # For expressions without derivativesÍ›
}

# Parse the L operator
L_terms = data["L"].split(" - ")  # Split terms by the "-" operator
L_expr = sum(differential_map[term] for term in L_terms if term in differential_map)
# Parse initial conditions
initial_conds = []
for cond in data["initial_conds"]:
    differential = differential_map.get(cond["differential"], None)
    expression = sp.sympify(cond["expression"])
    if differential is not None:
        initial_conds.append((differential, expression))
    else:
        print(f"Warning: Differential '{cond['differential']}' not recognized.")

# Parse boundary conditions
boundary_conds = []
for cond in data["boundary_conds"]:
    differential = differential_map.get(cond["differential"], None)
    expression = sp.sympify(cond["expression"])
    if differential is not None:
        boundary_conds.append((differential, expression))
    else:
        print(f"Warning: Differential '{cond['differential']}' not recognized.")

# Parse `u_0` and `u_g` as arrays for use in calculations
u_0 = np.array(data["u_0"])
u_g = np.array(data["u_g"])

# Display parsed information
print("L Operator:", L_expr)
print("Initial Conditions:", initial_conds)
print("Boundary Conditions:", boundary_conds)