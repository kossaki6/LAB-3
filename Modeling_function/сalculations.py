import sympy as sp
import numpy as np

#################################################################
                #TASK CONDITIONS EXTRACTION#
#################################################################
import json

def load_equations(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print("Successfully loaded JSON data.")
        return data
    except FileNotFoundError:
        print("File not found. Ensure the path is correct.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Load the data
data = load_equations('ideal_equations.json')

# Extract constraints
x1_min, x1_max = list(map(float, data.get('x1_constraints', '-2 2').split()))
x2_min, x2_max = list(map(float, data.get('x2_constraints', '-3 3').split()))
t_min, t_max = 0, float(data.get('T', 1))
x1, x2, t = sp.symbols('x1 x2 t')

# Extract functions and parameters
y_expression = data.get('y(s)', '')
u_expression = data.get('u(s)', '')

# Extract initial conditions
initial_conditions = data.get('initial_conds', [])
boundary_conditions = data.get('boundary_conds', [])

print(f"x1_constraints: {x1_min, x1_max}, x2_constraints: {x2_min, x2_max}, T: {t_max}")
print(f"y_expression: {y_expression}, u_expression: {u_expression}")
print(f"Initial Conditions: {initial_conditions}")
print(f"Boundary Conditions: {boundary_conditions}")

# Convert to SymPy expressions
y = sp.sympify(y_expression)
u = sp.sympify(u_expression)


'''# Define symbols
x1, x2, t = sp.symbols('x1 x2 t')
c = 1

# Constraints
x1_min, x1_max = -2, 2
x2_min, x2_max = -3, 3
t_min, t_max = 0, 1

# Define the y(s) function
y = x1 * t**2 + x2**2
print("function y:", y)

# Define the L operator
def L(y):
    # Compute the operator terms
    delta_t_2 = sp.Derivative(y, t, t)
    delta_x1_2 = sp.Derivative(y, x1, x1)
    delta_x2_2 = sp.Derivative(y, x2, x2)
    
    # Combine into L
    L = delta_t_2 - c**2*(delta_x1_2 + delta_x2_2)
    return L

# Define u(s)
u = sp.simplify(L(y))
print("function u:", u)'''



#################################################################
                        #BOUNDARY CONDITIONS#
#################################################################
'''# Operators for boundary conditions
def L_0_1(y):
    # Represents ∂y/∂t
    return sp.Derivative(y, t)
def L_0_2(y):
    # Represents y
    return y

def L_g_1(y):
    # Represents y
    return y
def L_g_2(y):
    # Represents ∂y/∂x1
    return sp.Derivative(y, x1)
def L_g_3(y):
    # Represents ∂y/∂x2
    return sp.Derivative(y, x2)'''

# Define Y_0_g(x)
def Y_0_g():
    # Parse initial conditions
    initial_terms = []
    for cond in data['initial_conds']:
        expression = sp.sympify(cond['expression'])  # Convert string to symbolic expression
        initial_terms.append(expression)

    # Parse boundary conditions
    boundary_terms = []
    for cond in data['boundary_conds']:
        expression = sp.sympify(cond['expression'])
        boundary_terms.append(expression)

    # Combine all terms
    return initial_terms + boundary_terms

print(f"Y_0_g:{Y_0_g()}")



#################################################################
                    #POINTS OF DISCRETIZATION#
#################################################################
# Defining step
step_net = 0.5
# Net for calculations
x1_vals = np.linspace(x1_min, x1_max, int((x1_max-x1_min)/step_net))
x2_vals = np.linspace(x2_min, x2_max, int((x2_max-x2_min)/step_net))
t_vals = np.linspace(t_min, t_max, int((t_max-t_min)/step_net))
# Generate the net
s_net = np.array([[x1, x2, t] for x1 in x1_vals for x2 in x2_vals for t in t_vals])

# Points for u, u_0 and u_g
# Define index points
num_discretization = 8
x1_points = np.linspace(x1_min, x1_max, num_discretization)
x2_points = np.linspace(x2_min, x2_max, num_discretization)
t_points = np.linspace(t_min, t_max, num_discretization)

# Generate S_m for u_m
s_m = np.array([[x1, x2, t] for x1 in x1_points for x2 in x2_points for t in t_points])

# Generate S_0 for u_0
t_lower = -1
t_0_points = np.linspace(t_lower, t_min, num_discretization)
s_0 = np.array([[x1, x2, t] for x1 in x1_points for x2 in x2_points for t in t_0_points])

# Generate S_g for u_g
step_g = 1
x1_outers = [x1_min - step_g, x1_max + step_g]
x2_outers = [x2_min - step_g, x2_max + step_g]

# Create the border without corners
border_points = [[x1, x2] for x1 in x1_outers for x2 in x2_points] +\
    [[x1, x2] for x1 in x1_points for x2 in x2_outers]

# t values for the area
t_g_points = t_points[1:]

s_g = np.array([[x1, x2, t] for x1, x2 in border_points for t in t_g_points])
# print("Discretization for u_m", s_m, "Discretization for u_0", s_0, "Discretization for u_g", s_g, sep='\n')

M0 = len(s_0)
Mg = len(s_g)
M = M0+Mg
# Creating a united set of points
s_0_g_combined = np.concatenate((s_0, s_g))


#################################################################
                    #TURNING FUNCTIONS NUMERICAL#
#################################################################
# Numerical derivative
def num_derivative(function, position_of_var, point):
    h = max(1e-5, 1e-8 * abs(point[position_of_var]))
    h_point_plus = point.copy()
    h_point_minus = point.copy()
    h_point_plus[position_of_var] += h
    h_point_minus[position_of_var] -= h
    return (function(h_point_plus) - function(h_point_minus)) / (2 * h)

# Redefine u_s
def u_s(s):
    x1_val, x2_val, t_val = s
    return u.subs({x1: x1_val, x2: x2_val, t:t_val})

# Redefine y_s
def y_s(s):
    x1_val, x2_val, t_val = s
    return y.subs({x1: x1_val, x2: x2_val, t:t_val})

# Redefine Y_0_g_s
def Y_0_g_s(i, s):
    x1_val, x2_val, t_val = s
    return Y_0_g()[i].subs({x1: x1_val, x2: x2_val, t:t_val})

# Operators for boundary conditions
def L_0_1_s(y, s):
    # Represents ∂y/∂t
    return num_derivative(y,2,s)
def L_0_2_s(y, s):
    # Represents y
    return y(s)

def L_g_1_s(y, s):
    # Represents y
    return y(s)
def L_g_2_s(y, s):
    # Represents ∂y/∂x1
    return num_derivative(y,0,s)
def L_g_3_s(y, s):
    # Represents ∂y/∂x2
    return num_derivative(y,1,s)


# List of operators
initial_operators = [
    L_0_1_s,
    L_0_2_s
]
boundary_operators = [ 
    L_g_1_s,
    L_g_2_s,
    L_g_3_s
]



#################################################################
                        #GREEN'S FUNCTION#
#################################################################
# Define the Green's function G(s)
def G_function(s):
    x1, x2, t = s
    r = np.sqrt(x1**2 + x2**2)
    c = 1
    sqrt_term = c**2 * t**2 - r**2
    if sqrt_term <= 0:
        return 0
    return (t - r / c) / (2 * np.pi * np.sqrt(sqrt_term))



#################################################################
                    #CALCULATING y infinity#
#################################################################
# Compute u_m at s_m
u_m = []
for s in s_m:
    u_val = u_s(s)
    u_m.append(u_val)
# print("u_m:", u_m)

# Compute y_inf(s) at desired points
def y_inf(s):
    y_inf_val = 0
    for m, s_point in enumerate(s_m):
        G_val = G_function(s - s_point)
        y_inf_val += G_val * u_m[m]
    return y_inf_val

'''print("y_infinity:")
for s in s_0_g_combined:
    print(y_inf(s), end=', ')'''



#################################################################
                #CALCULATING B(s) AND Y(s) FOR EACH s#
#################################################################

# Build B matrix and Y vector
def calculate_B_and_Y(s):
    B_holder = []
    Y_holder = []
    # Loop over each boundary operator
    for i, L_op in enumerate(initial_operators + boundary_operators):
        row = []
        for s_point in s_0_g_combined:
            # Apply operator L_op to G_val at s-s_point
            L_G_val = L_op(G_function,s - s_point)
            row.append(L_G_val)
        B_holder.append(row)

        Y_holder.append(Y_0_g_s(i,s) - L_op(y_inf, s))    
    return np.array(B_holder), np.array(Y_holder).T



#################################################################
                        #CALCULATING u#
#################################################################


P = np.zeros((M, M), dtype=np.float64)
B_y = np.zeros((M, ), dtype=np.float64)

# Numerical approximation of P integral
for s in s_net:
    B_s, Y_s = calculate_B_and_Y(s)
            
    # Accumulate contributions to P and B_y
    P_temp = B_s.T @ B_s
    P_temp = np.array(P_temp, dtype=np.float64)
    B_y_temp = B_s.T @ Y_s
    B_y_temp = np.array(B_y_temp, dtype=np.float64)

    P += P_temp
    B_y += B_y_temp

# Compute Pseudo-inverse of P
P_pseudo_inverse = np.linalg.pinv(P)

# Calculate u
u_solution = P_pseudo_inverse @ B_y



#################################################################
                        #CALCULATING y 0#
#################################################################
# Separating the u_0
u_0 = u_solution[:M0]

def y_0(s):
    y0_val = 0
    for m, s_point in enumerate(s_0):
        G_val = G_function(s - s_point)
        y0_val += G_val * u_0[m]
    return y0_val



#################################################################
                        #CALCULATING y g#
#################################################################
# Separating the u_g
u_g = u_solution[M0:]

def y_g(s):
    yg_val = 0
    for m, s_point in enumerate(s_g):
        G_val = G_function(s - s_point)
        yg_val += G_val * u_g[m]
    return yg_val



#################################################################
                        #CALCULATING y total#
#################################################################
def y_total(s):
    return y_inf(s) + y_0(s) + y_g(s)



#################################################################
                        #VISUALIZATION AND ANALYSIS#
#################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
x1_flat = x1_grid.flatten()
x2_flat = x2_grid.flatten()

# Initial time value
t_fixed = 0.8
original_function_flat = [y_s([x1, x2, t_fixed]) for x1, x2 in zip(x1_flat, x2_flat)]
found_solution_flat = [y_total([x1, x2, t_fixed]) for x1, x2 in zip(x1_flat, x2_flat)]

# Reshape the results back to grid format
original_function = np.array(original_function_flat).reshape(x1_grid.shape)
found_solution = np.array(found_solution_flat).reshape(x1_grid.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the original function
surf_original = ax.plot_surface(
    x1_grid, x2_grid, original_function, cmap='viridis', edgecolor='none', alpha=0.7, label='Original'
)

# Plot the found solution
surf_found = ax.plot_surface(
    x1_grid, x2_grid, found_solution, cmap='plasma', edgecolor='none', alpha=0.7, label='Found Solution'
)

# Add labels and title
ax.set_title(f"3D Plot at t = {t_fixed}", fontsize=14)
ax.set_xlabel("x1", fontsize=12)
ax.set_ylabel("x2", fontsize=12)
ax.set_zlabel("y", fontsize=12)

plt.show()