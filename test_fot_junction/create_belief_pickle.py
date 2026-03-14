import pickle

# -----------------------------
# Intent sets for two vehicles
# -----------------------------
Lambda_1 = ["lanekeep", "leftturn"]
Lambda_2 = ["leftturn", "yield"]

# Cartesian product of intentions
Theta = [(i, j) for i in Lambda_1 for j in Lambda_2]

print("Joint intent space Θ:")
for t in Theta:
    print(t)

# -----------------------------
# Uniform initial belief
# -----------------------------
num_states = len(Theta)
uniform_prob = 1.0 / num_states

'''
('lanekeep', 'leftturn')
('lanekeep', 'yield')
('leftturn', 'leftturn')
('leftturn', 'yield')
'''

belief = [uniform_prob] * num_states

print("\nInitial belief:", belief)

# -----------------------------
# Because planner uses belief[7:]
# we prepend 7 dummy values
# -----------------------------
belief_vector = [0.0]*7 + belief

# -----------------------------
# Save to pickle file
# -----------------------------
file_name = "belief_updater.pickle"

with open(file_name, "wb") as f:
    pickle.dump(belief_vector, f)

print(f"\nPickle file '{file_name}' created successfully.")
print("Stored vector:", belief_vector)
