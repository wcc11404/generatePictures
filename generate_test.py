import numpy as np
import dcgan_mnist
import dcgan
import tqdm

# DC=dcgan_mnist.DCGAN()
# DC.trainModelLoop()
# DC.testModel(isLoadModel=True)
DC1=dcgan.DCGAN()
#DC1.trainModelLoop()
DC1.trainModel(100,isShowMessage=True,isSaveModel=True)
DC1.testModel(isLoadModel=True)