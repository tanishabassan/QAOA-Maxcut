# useful additional packages
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

def obj_funct(Q_program, pauli_list, entangler_map, coupling_map, initial_layout, n, m, backend, shots, theta):
    """ Evaluate the objective function for a classical optimization problem.

    Q_program is an instance object of the class quantum program
    pauli_list defines the cost function as list of ising terms with weights
    theta are the control parameters
    n is the number of qubits
    m is the depth of the trial function
    backend is the type of backend to run it on
    shots is the number of shots to run. Taking shots = 1 only works in simulation
    and computes an exact average of the cost function on the quantum state
    """
    std_cost=0 # to add later
    circuits = ['trial_circuit']


    if shots==1:
        Q_program.add_circuit('trial_circuit', trial_circuit_ry(n, m, theta, entangler_map, None, False))
        result = Q_program.execute(circuits, backend=backend, coupling_map=coupling_map, initial_layout=initial_layout, shots=shots)
        state = result.get_data('trial_circuit')['quantum_state']
        cost=Energy_Estimate_Exact(state,pauli_list,True)

    else:
        Q_program.add_circuit('trial_circuit', trial_circuit_ry(n, m, theta, entangler_map, None, True))
        result = Q_program.execute(circuits, backend=backend, coupling_map=coupling_map, initial_layout=initial_layout, shots=shots)
        data = result.get_counts('trial_circuit')
        cost = Energy_Estimate(data, pauli_list)



    return cost, std_cost



#######################################################################################

# Generating a graph of 4 nodes

n =4 # Number of nodes in graph

G=nx.Graph()
G.add_nodes_from(np.arange(0,n,1))
elist=[(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]
# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(elist)

colors = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
default_axes.set_xlim(-0.1,1.1)
default_axes.set_ylim(-0.1,1.1)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes)

# Computing the weight matrix from the random graph

w = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i,j,default=0)
        if temp != 0:
            w[i,j] = temp['weight']

        print(w)

#######################################################################################


best_cost_brute = 0
for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + w[i,j]*x[i]*(1-x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x

    print('case = ' + str(x)+ ' cost = ' + str(cost))

colors = []
for i in range(n):
    if xbest_brute[i] == 0:
        colors.append('r')
    else:
        colors.append('b')
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8)
plt.show()
print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))


# Determining the constant shift and initialize a pauli_list that contains the ZZ Ising terms

pauli_list = []
cost_shift = 0
for i in range(n):
    for j in range(i):
        if w[i,j] != 0:
            cost_shift = cost_shift + w[i,j]
            wp = np.zeros(n)
            vp = np.zeros(n)
            vp[n-i-1] = 1
            vp[n-j-1] = 1
            pauli_list.append((w[i,j],Pauli(vp,wp)))
cost_shift


#Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector

H = make_Hamiltonian(pauli_list)
we, ve = la.eigh(H, eigvals=(0, 1))
exact = we[0]
exact_maxcut = -we[0]/2+cost_shift/2
print(exact_maxcut)
print(exact)
H = np.diag(H)

from qiskit import QuantumProgram
# Creating Programs create your first QuantumProgram object instance.
Q_program = QuantumProgram()

# Set your API Token
# You can get it from https://quantumexperience.ng.bluemix.net/qx/account,
# looking for "Personal Access Token" section.
QX_TOKEN = '048357f58639d4c6788b6ae75478a7ba5b3ad761aee7773121d7f1a16eaa685d1bbd27b25d3e5dec6b22aca732ebe61a28b632a9938baf4762bb743857cc12fe'
QX_URL = 'https://quantumexperience.ng.bluemix.net/api'


# Set up the API and execute the program.
# You need the API Token and the QX URL.
Q_program.set_api(QX_TOKEN, QX_URL)



#In [19]:
# Testing Optimization on a quantum computer

# Quantum circuit parameters:

# the entangler step is made of two-qubit gates between a control and target qubit, control: [target]
entangler_map = {0: [1], 1: [2], 2: [3]}

# the coupling_maps gates allowed on the device
coupling_map = None
# the layout of the qubits
initial_layout = None

# the backend used for the quantum computation
backend = 'local_qasm_simulator'
# Total number of trial steps used in the optimization
max_trials = 100;
n = 4 # the number of qubits
# Depth of the quantum circuit that prepares the trial state
m = 3
# initial starting point for the control angles
initial_theta=np.random.randn(m*n)
# number of shots for each evaluation of the cost function (shots=1 corresponds to perfect evaluation,
# only available on the simulator)
shots = 1
# choose to plot the results of the optimizations every save_steps
save_step = 1


""" ##########################      RUN    OPTIMIZATION      #######################
if shots == 1:
    obj_funct_partial = partial(obj_funct, Q_program, pauli_list, entangler_map, coupling_map, initial_layout, n, m, backend, shots)
    initial_c=0.01
else:
    obj_funct_partial = partial(obj_funct, Q_program, pauli_list, entangler_map, coupling_map, initial_layout, n, m, backend, shots)
    initial_c=0.1

target_update=2*np.pi*0.1
SPSA_parameters=SPSA_calibration(obj_funct_partial,initial_theta,initial_c,target_update,25)
print ('SPSA parameters = ' + str(SPSA_parameters))

best_distance_quantum, best_theta, cost_plus, cost_minus,_,_ = SPSA_optimization(obj_funct_partial, initial_theta, SPSA_parameters, max_trials, save_step)


"""

def cost_function(Q_program,H,n,m,entangler_map,shots,device,theta):

    return eval_hamiltonian(Q_program,H,trial_circuit_ry(n,m,theta,entangler_map,None,False),shots,device).real


initial_c=0.1
target_update=2*np.pi*0.1
save_step = 1

if shots !=1:
    H=group_paulis(pauli_list)

SPSA_params = SPSA_calibration(partial(cost_function,Q_program,H,n,m,entangler_map,
                                           shots,backend),initial_theta,initial_c,target_update,25)

best_distance_quantum, best_theta, cost_plus, cost_minus, _, _ = SPSA_optimization(partial(cost_function,Q_program,H,n,m,entangler_map,shots,backend),
                                                           initial_theta,SPSA_params,max_trials,save_step,1);




plt.plot(np.arange(0, max_trials,save_step), cost_plus,label='C(theta_plus)')
plt.plot(np.arange(0, max_trials,save_step),cost_minus,label='C(theta_minus)')
plt.plot(np.arange(0, max_trials,save_step),np.ones(max_trials//save_step)*best_distance_quantum, label='Final Optimized Cost')
plt.plot(np.arange(0, max_trials,save_step),np.ones(max_trials//save_step)*exact, label='Exact Cost')
plt.legend()
plt.xlabel('Number of trials')
plt.ylabel('Cost')

shots = 5000
circuits = ['final_circuit']
Q_program.add_circuit('final_circuit', trial_circuit_ry(n, m, best_theta, entangler_map, None, True))
result = Q_program.execute(circuits, backend=backend, shots=shots, coupling_map=coupling_map, initial_layout=initial_layout)
data = result.get_counts('final_circuit')
plot_histogram(data,5)




# Getting the solution and cost from the largest component of the optimal quantum state

max_value = max(data.values())  # maximum value
max_keys = [k for k, v in data.items() if v == max_value] # getting all keys containing the `maximum`

x_quantum=np.zeros(n)
for bit in range(n):
    if max_keys[0][bit]=='1':
        x_quantum[bit]=1

best_cost_quantum = 0
for i in range(n):
    for j in range(n):
        best_cost_quantum+= w[i,j]*x_quantum[i]*(1-x_quantum[j])


# Plot the quantum solution
colors = []
for i in range(n):
    if x_quantum[i] == 0:
        colors.append('r')
    else:
        colors.append('b')
nx.draw_networkx(G, node_color=colors, node_size=600, alpha = .8)

print('Best solution from the quantum optimization is = ' +str(x_quantum)+ ' with cost = ' + str(best_cost_quantum))
