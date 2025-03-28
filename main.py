import pickle

# Load the pickle file
with open('medicine_dict.pkl', 'rb') as f:
    data = pickle.load(f)

# View the contents
print(data)
