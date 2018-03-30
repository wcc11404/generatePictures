import numpy as np
from dcgan_mnist import *
import tqdm

DC=DCGAN()
# DC.trainModelLoop()
DC.testModel(isLoadModel=True)