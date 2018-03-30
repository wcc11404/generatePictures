import numpy as np
from dcgan import *
import tqdm

DC=DCGAN()
DC.trainModelLoop()
#DC.testModel(isLoadModel=True)