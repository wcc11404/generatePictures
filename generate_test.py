import numpy as np

def make_one_hot(data1):
    return (np.arange(10)==data1[:,None]).astype(np.integer)

y=np.array([1,2])
z=make_one_hot(y)

print(z)
sample_y=np.random.randint(0,9,64)
print(sample_y)