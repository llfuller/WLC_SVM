import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

file2DArray = np.transpose(np.loadtxt('MixedOdorData_Combo0.txt'))
listOfAlpha, listOfBeta, odor1Proportion, FitOdor1Array_Ordered = file2DArray
print(np.shape(listOfAlpha))
#should be 210 long

fig = plt.figure()
figSubplot = fig.add_subplot(111, projection='3d')
# Plotting proportion result:
figSubplot.scatter(listOfAlpha, listOfBeta, odor1Proportion, color="r", label="Amount in odor 1")
figSubplot.set_xlabel('Alpha')
figSubplot.set_ylabel('Beta')
# Plotting softmax fit:
figSubplot.scatter(listOfAlpha, listOfBeta, FitOdor1Array_Ordered,color='b', label="Softmax fit to odor1")
plt.legend()
plt.show()