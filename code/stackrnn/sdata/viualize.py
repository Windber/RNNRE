import pickle
f = open('epoch2', 'rb')
p = pickle.load(f)
b = p[0]
print(b.shape)