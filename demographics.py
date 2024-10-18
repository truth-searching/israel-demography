import numpy as np
import matplotlib.pyplot as plt

def simulate_population_growth(transition_matrix, birth_rates, mortality_rate, initial_sizes, years=50):
    """
    Simulate population growth over a given number of years.
    
    Parameters:
    - transition_matrix: A square matrix where entry (i, j) represents the transition rate from group i to group j.
    - birth_rates: A vector where entry i represents the birth rate for group i.
    - initial_sizes: A vector where entry i represents the initial size of group i.
    - years: The number of years to simulate (default 50).
    
    Returns:
    - A dictionary containing population sizes for each group over time.
    """
    num_groups = len(initial_sizes)
    populations = np.zeros((years + 1, num_groups))
    populations[0] = initial_sizes

    # Simulate for each year
    for t in range(1, years + 1):
        # Apply transition matrix to update group sizes
        #populations[t] = np.dot(transition_matrix, populations[t - 1])
        #populations[t] = np.dot(populations[t - 1], transition_matrix)
        populations[t] = np.matmul(transition_matrix, populations[t - 1])
        #print(np.dot(transition_matrix[0,:], populations[t-1]), populations[t][0], transition_matrix[0,:])
        # Apply birth rates
        populations[t] = populations[t] * birth_rates * mortality_rate
    
    return populations

# Example parameters
# Transition matrix (rows: from group, columns: to group)
# the table is based on https://www.haaretz.co.il/magazine/2024-07-04/ty-article-magazine/.highlight/00000190-7896-d14c-a1dd-fb9f6ec40000
transition_matrix = np.array([ # actual
    [0.75, 0.38, 0.09, 0.07, 0.03, 0.06, 0.01, 0, 0],
    [0.15, 0.42, 0.27, 0.14, 0.09, 0.07, 0.04, 0, 0],
    [0.03, 0.12, 0.47, 0.31, 0.19, 0.07, 0.05, 0, 0],
    [0.02, 0.02, 0.05, 0.34, 0.09, 0.08, 0.01, 0, 0],
    [0.01, 0.03, 0.07, 0.08, 0.40, 0.16, 0.05, 0, 0],
    [0.02, 0.02, 0.02, 0.03, 0.15, 0.46, 0.07, 0, 0],
    [0.02, 0.01, 0.03, 0.03, 0.05, 0.10, 0.77, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])
transition_matrix_III = np.array([ # haredi same as hardal
    [0.75, 0.38, 0.09, 0.07, 0.03, 0.06, 0.06, 0],
    [0.15, 0.42, 0.27, 0.14, 0.09, 0.07, 0.07, 0],
    [0.03, 0.12, 0.47, 0.31, 0.19, 0.07, 0.07, 0],
    [0.02, 0.02, 0.05, 0.34, 0.09, 0.08, 0.08, 0],
    [0.01, 0.03, 0.07, 0.08, 0.40, 0.16, 0.10, 0],
    [0.02, 0.02, 0.02, 0.03, 0.15, 0.46, 0.16, 0],
    [0.02, 0.01, 0.03, 0.03, 0.05, 0.10, 0.46, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

#transition_matrix = transition_matrix_III

transition_matrix_II = np.array([ # worst
    [0.75, 0.38, 0.09, 0.07, 0.03, 0.06, 0.01, 0],
    [0.15, 0.42, 0.27, 0.14, 0.09, 0.07, 0.01, 0],
    [0.03, 0.12, 0.47, 0.31, 0.19, 0.07, 0.02, 0],
    [0.02, 0.02, 0.05, 0.34, 0.09, 0.08, 0.02, 0],
    [0.01, 0.03, 0.07, 0.08, 0.40, 0.16, 0.05, 0],
    [0.02, 0.02, 0.02, 0.03, 0.15, 0.46, 0.07, 0],
    [0.02, 0.01, 0.03, 0.03, 0.05, 0.10, 0.82, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])
transition_matrix_I = np.array([ # desired
    [0.77, 0.4, 0.12, 0.1, 0.08, 0.16, 0.99, 0],
    [0.15, 0.42, 0.27, 0.14, 0.09, 0.07, 0.00, 0],
    [0.03, 0.11, 0.47, 0.31, 0.19, 0.07, 0.00, 0],
    [0.02, 0.02, 0.05, 0.34, 0.09, 0.12, 0.00, 0],
    [0.01, 0.03, 0.07, 0.08, 0.44, 0.18, 0.00, 0],
    [0.02, 0.02, 0.02, 0.03, 0.11, 0.40, 0.00, 0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

# debugging case (secular rules):
#transition_matrix = np.eye(8)
#transition_matrix[0,1:7] = 0.99
#for i in range(1,7):
#    transition_matrix[i,i] = 0.01

print(transition_matrix)

s = np.sum(transition_matrix, axis=0)
print(s)
assert abs(max(s)-min(s)) < 1e-9

conv_year = 18
for i in range(transition_matrix.shape[0]):
    transition_matrix[i,i] = 1 - (1 - transition_matrix[i,i])/conv_year
    for j in range(transition_matrix.shape[0]):
        if i != j:
            transition_matrix[i,j] = transition_matrix[i,j]/conv_year

print(transition_matrix)

# childern for a woman: based in wikipedia
children = np.array([1.96, 2, 2.3, 2.6, 3.92, 5, 6.64, 3.01, 4])

# Birth rates vector (example values)
birth_rates = np.array([1.021, 1.030, 1.040, 1.050, 1.060, 1.068, 1.070])
birth_rates = 1 + 5 * children / 1000
print('birth_rates', birth_rates)
mortality_rate = 1 - 6 / 1000

group_labels = ['Secular', 'Secular-Traditional', 'Traditional', 'Dati-Liberal', 'Dati-Leumi', 'Hardal', 'Haredi', 'Non Jews', 'Occupied Arabs']

# Initial group sizes (example values): based on the elections for the Kneset and poles
non_jews_prop = 0.264  # 0.5  # 0.264 0.181 christians = 0.013 druze = 0.015 muslems = 
occupied_arabs = 0.25
jews_prop = 1 - (non_jews_prop + occupied_arabs)
inter_jews_prop = [0.1775, 0.2, 0.1682, 0.0844, 0.1543, 0.038, 0.1776]
initial_prop = np.array([p*jews_prop for p in inter_jews_prop] + [non_jews_prop, occupied_arabs])
initial_sizes = 10e7 * initial_prop
print(sum(initial_prop))
assert abs(sum(initial_prop) - 1) < 1e-9
assert abs(sum(inter_jews_prop) - 1) < 1e-9

# Simulate population growth for 50 years
y=100

populations = simulate_population_growth(transition_matrix, birth_rates, mortality_rate, initial_sizes, years=y)

# Plot the results
years = np.arange(y+1)

plt.figure(figsize=(10, 6))

liberal = populations[:,0]+populations[:,1]+populations[:,3] #+populations[:,7]
non_liberal = populations[:,2]+populations[:,4]+populations[:,5]+populations[:,6]

plt.figure(1)
for i, label in enumerate(group_labels):
    plt.plot(years, populations[:, i]/np.sum(populations, axis=1), label=label)
#    #plt.plot(years, populations[:, i], label=label)
plt.title('Population Growth Over 100 Years')
plt.xlabel('Years')
plt.ylabel('Population Size')
plt.legend()
plt.grid(True)
plt.savefig('population_tribes_100_years.png')

plt.figure(2)
plt.plot(years, liberal/np.sum(populations, axis=1), label='liberal')
plt.plot(years, non_liberal/np.sum(populations, axis=1), label='non liberal')
plt.plot(years, populations[:,7]/np.sum(populations, axis=1), label='non jews')
plt.plot(years, populations[:,8]/np.sum(populations, axis=1), label='occupied arabs')

plt.title('Population Growth Over 100 Years')
plt.xlabel('Years')
plt.ylabel('Population Size')
plt.legend()
plt.grid(True)
plt.savefig('coalitions_100_years.png')

plt.show()
