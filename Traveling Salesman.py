import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
import matplotlib.axes as axes

import numpy as np
from scipy import linalg as la
from itertools import permutations
from functools import partial
import networkx as nx

# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ry, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Energy_Estimate, make_Hamiltonian, eval_hamiltonian, group_paulis
from qiskit.tools.qi.pauli import Pauli

# Random choice of the cities/nodes
N = 4
xc = (np.random.rand(N)-0.5)*10
yc = (np.random.rand(N)-0.5)*10

plt.scatter(xc, yc, s=200)
for i in range(len(xc)):
    plt.annotate(i,(xc[i]+0.15,yc[i]),size=16,color='r')
plt.show()

# Getting the distances
w = np.zeros([N,N])
for i in range(N):
    for j in range(N):
        w[i,j]= np.sqrt((xc[i]-xc[j])**2+(yc[i]-yc[j])**2)


a=list(permutations(range(1,N)))
last_best_distance = 10000000
for i in a:
    distance = 0
    pre_j = 0
    for j in i:
        distance = distance + w[j,pre_j]
        pre_j = j
    distance = distance + w[0,pre_j]
    order = (0,) + i
    if distance < last_best_distance:
        best_order = order
        last_best_distance = distance
    print('order = ' + str(order) + ' Distance = ' + str(distance))

best_distance_brute = last_best_distance
best_order_brute = best_order

plt.scatter(xc, yc)
xbest = np.array([xc[i] for i in best_order_brute])
xbest = np.append(xbest,xbest[0])
ybest = np.array([yc[i] for i in best_order_brute])
ybest = np.append(ybest,ybest[0])
plt.plot(xbest, ybest, 'b.-', ms = 40)
plt.plot(xc[0], yc[0], 'r*', ms = 20)
for i in range(len(xc)):
    plt.annotate(i,(xc[i]+0.2,yc[i]),size=16,color='r')
plt.show()
print('Best order from brute force = ' + str(best_order_brute) + ' with total distance = ' + str(best_distance_brute))

n=(N-1)**2 # number of qubits
A = np.max(w)*100 # A parameter of cost function

# takes the part of w matrix excluding the 0-th point, which is the starting one
wsave = w[1:N,1:N]
# nearest-neighbor interaction matrix for the prospective cycle (p,p+1 interaction)
shift = np.zeros([N-1,N-1])
shift = la.toeplitz([0,1,0], [0,1,0])/2

# the first and last point of the TSP problem are fixed by initial and final conditions
firststep = np.zeros([N-1])
firststep[0] = 1;
laststep = np.zeros([N-1])
laststep[N-2] = 1;

# The binary variables that define a path live in a tensor product space of position and ordering indices

# Q defines the interactions between variables
Q = np.kron(shift,wsave) + np.kron(A*np.ones((N-1, N-1)), np.identity(N-1)) + np.kron(np.identity(N-1),A*np.ones((N-1, N-1)))
# G defines the contribution from the individual variables
G = np.kron(firststep,w[0,1:N]) + np.kron(laststep,w[1:N,0]) - 4*A*np.kron(np.ones(N-1),np.ones(N-1))
# M is the constant offset
M = 2*A*(N-1)

# Evaluates the cost distance from a binary representation of a path
fun = lambda x: np.dot(np.around(x),np.dot(Q,np.around(x)))+np.dot(G,np.around(x))+M

def get_order_tsp(x):
    # This function takes in a TSP state, an array of (N-1)^2 binary variables, and returns the
    # corresponding travelling path associated to it
    order = [0]
    for p in range(N-1):
        for j in range(N-1):
            if x[(N-1)*p+j]==1:
                order.append(j+1)
    return order

def get_x_tsp(order):
    # This function takes in a traveling path and returns a TSP state, in the form of an array of (N-1)^2
    # binary variables
    x = np.zeros((len(order)-1)**2)
    for j in range(1,len(order)):
        p=order[j]
        x[(N-1)*(j-1)+(p-1)]=1
    return x


# Checking if the best results from the brute force approach are correct for the mapped system of binary variables

# Conversion from a path to a binary variable array
xopt_brute = get_x_tsp(best_order_brute)

print('Best path from brute force mapped to binary variables: \n')
print(xopt_brute)

flag = False
for i in range(100000):
    rd = np.random.randint(2, size=n)
    if fun(rd) < (best_distance_brute - 0.0001):
        print('\n A random solution is better than the brute-force one. The path measures')
        print(fun(rd))
        flag = True

if flag == False:
    print('\nCheck with 10^5 random solutions: the brute-force solution mapped to binary variables is correct.\n')

print('Shortest path evaluated with binary variables: ')
print(fun(xopt_brute))

# Optimization with simulated annealing

initial_x = np.random.randint(2, size=n)

cost = fun(initial_x)
x = np.copy(initial_x)
alpha = 0.999
temp = 10
for j in range(10000):

    # pick a random index and flip the bit associated with it
    flip = np.random.randint(len(x))
    new_x = np.copy(x)
    new_x[flip] = (x[flip] + 1) % 2

    # compute cost function with flipped bit
    new_cost = fun(new_x)
    if np.exp(-(new_cost - cost) / temp) > np.random.rand():
        x = np.copy(new_x)
        cost = new_cost
    temp = temp * alpha
print('distance = ' + str(cost) + ' x_solution = ' + str(x) + ', final temperature= ' + str(temp))

best_order_sim_ann = get_order_tsp(x)

plt.scatter(xc, yc)
xbest = np.array([xc[i] for i in best_order_sim_ann])
xbest = np.append(xbest, xbest[0])
ybest = np.array([yc[i] for i in best_order_sim_ann])
ybest = np.append(ybest, ybest[0])
plt.plot(xbest, ybest, 'b.-', ms=40)
plt.plot(xc[0], yc[0], 'r*', ms=20)
for i in range(len(xc)):
    plt.annotate(i, (xc[i] + 0.15, yc[i]), size=16, color='r')
plt.show()
print('Best order from simulated annealing = ' + str(best_order_sim_ann) + ' with total distance = ' + str(cost))


# Defining the new matrices in the Z-basis

Iv=np.ones((N-1)**2)
Qz = (Q/4)
Gz =( -G/2-np.dot(Iv,Q/4)-np.dot(Q/4,Iv))
Mz = (M+np.dot(G/2,Iv)+np.dot(Iv,np.dot(Q/4,Iv)))

Mz = Mz + np.trace(Qz)
Qz = Qz - np.diag(np.diag(Qz))

# Recall the change of variables is
# x = (1-z)/2
# z = -2x+1
z= -(2*xopt_brute)+Iv

for i in range(1000):
    rd =  1-2*np.random.randint(2, size=n)
    if np.dot(rd,np.dot(Qz,rd))+np.dot(Gz,rd)+Mz < (best_distance_brute-0.0001):
        print(np.dot(rd,np.dot(Qz,rd))+np.dot(Gz,rd)+Mz)

# Getting the Hamiltonian in the form of a list of Pauli terms

pauli_list = []
for i in range(n):
    if Gz[i] != 0:
        wp = np.zeros(n)
        vp = np.zeros(n)
        vp[i] = 1
        pauli_list.append((Gz[i], Pauli(vp, wp)))
for i in range(n):
    for j in range(i):
        if Qz[i, j] != 0:
            wp = np.zeros(n)
            vp = np.zeros(n)
            vp[i] = 1
            vp[j] = 1
            pauli_list.append((2 * Qz[i, j], Pauli(vp, wp)))

pauli_list.append((Mz, Pauli(np.zeros(n), np.zeros(n))))
# Making the Hamiltonian as a full matrix and finding its lowest eigenvalue

H = make_Hamiltonian(pauli_list)
we, v = la.eigh(H, eigvals=(0, 0))
exact = we[0]
print(exact)
H = np.diag(H)

# Setting up a quantum program and connecting to the Quantum Experience API
Q_program = QuantumProgram()
# set the APIToken and API url
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url'])
# Optimization of the TSP using a quantum computer

# Quantum circuit parameters

# the entangler step is made of two-qubit gates between a control and target qubit, control: [target]
coupling_map = None

# the coupling_maps gates allowed on the device
entangler_map = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8]}
# the layout of the qubits
initial_layout = None

# the backend used for the quantum computation
backend = 'local_qasm_simulator'
# Total number of trial steps used in the optimization
max_trials = 1500;
n = 9  # the number of qubits
# Depth of the quantum circuit that prepares the trial state
m = 5
# initial starting point for the control angles
initial_theta = np.random.randn(m * n)
# number of shots for each evaluation of the cost function (shots=1 corresponds to perfect evaluation,
# only available on the simulator)
shots = 1
# choose to plot the results of the optimizations every save_steps
save_step = 1

""" ##########################      RUN OPTIMIZATION      #######################

if shots == 1:
    obj_funct_partial = partial(obj_funct, Q_program, pauli_list, entangler_map, coupling_map, initial_layout, n, m, backend, shots)
    initial_c=0.01
else:
    obj_funct_partial = partial(obj_funct, Q_program, pauli_list, entangler_map, coupling_map, initial_layout, n, m, backenddevice, shots)
    initial_c=0.1


target_update=2*np.pi*0.1
SPSA_parameters=SPSA_calibration(obj_funct_partial,initial_theta,initial_c,target_update,25)
print ('SPSA parameters = ' + str(SPSA_parameters))    

best_distance_quantum, best_theta, cost_plus, cost_minus,_,_ = SPSA_optimization(obj_funct_partial, initial_theta, SPSA_parameters, max_trials, save_step)
"""


def cost_function(Q_program, H, n, m, entangler_map, shots, device, theta):
    return eval_hamiltonian(Q_program, H, trial_circuit_ry(n, m, theta, entangler_map, None, False), shots, device).real


initial_c = 0.1
target_update = 2 * np.pi * 0.1
save_step = 1

if shots != 1:
    H = group_paulis(pauli_list)

SPSA_params = SPSA_calibration(partial(cost_function, Q_program, H, n, m, entangler_map,
                                       shots, backend), initial_theta, initial_c, target_update, 25)

best_distance_quantum, best_theta, cost_plus, cost_minus, _, _ = SPSA_optimization(
    partial(cost_function, Q_program, H, n, m, entangler_map, shots, backend),
    initial_theta, SPSA_params, max_trials, save_step,1);

""" ##########################       PLOT RESULTS         #######################"""

plt.plot(np.arange(0, max_trials,save_step),cost_plus,label='C(theta_plus)')
plt.plot(np.arange(0, max_trials,save_step),cost_minus,label='C(theta_minus)')
plt.plot(np.arange(0, max_trials,save_step),(np.ones(max_trials//save_step)*best_distance_quantum), label='Final Cost')
plt.plot(np.arange(0, max_trials,save_step),np.ones(max_trials//save_step)*exact, label='Exact Cost')
plt.legend()
plt.xlabel('Number of trials')
plt.ylabel('Cost')


# Sampling from the quantum state generated with the optimal angles from the quantum optimization

shots = 100
circuits = ['final_circuit']
Q_program.add_circuit('final_circuit', trial_circuit_ry(n, m, best_theta, entangler_map,None,True))
result = Q_program.execute(circuits, backend=backend, shots=shots, coupling_map=coupling_map, initial_layout=initial_layout)
data = result.get_counts('final_circuit')
plot_histogram(data,5)

# Getting path and total distance from the largest component of the quantum state

max_value = max(data.values())  # maximum value
max_keys = [k for k, v in data.items() if v == max_value]  # getting all keys containing the `maximum`

x_quantum = np.zeros(n)
for bit in range(n):
    if max_keys[0][bit] == '1':
        x_quantum[bit] = 1

quantum_order = get_order_tsp(list(map(int, x_quantum)))
best_distance_quantum_amp = fun(x_quantum)
plt.scatter(xc, yc)
xbest = np.array([xc[i] for i in quantum_order])
xbest = np.append(xbest, xbest[0])
ybest = np.array([yc[i] for i in quantum_order])
ybest = np.append(ybest, ybest[0])
plt.plot(xbest, ybest, 'b.-', ms=40)
plt.plot(xc[0], yc[0], 'r*', ms=20)
for i in range(len(xc)):
    plt.annotate(i, (xc[i] + 0.15, yc[i]), size=14, color='r')
plt.show()
print('Best order from quantum optimization is = ' + str(quantum_order) + ' with total distance = ' + str(
    best_distance_quantum_amp))
