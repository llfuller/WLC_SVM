from __future__ import division

from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

defaultclock.dt = .05*ms

np.random.seed(125)

N_AL = 1000 #number of neurons in network
in_AL = .1 #inhibition g_nt in paper
PAL = 0.5 #probability of connection

#folders in which to save training and testing data
tr_prefix = 'train/'
te_prefix = 'test/test_'

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo, 
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V']
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net) #creates neuron group, synapses, spike and state monitors, etc

net.store() # saves current unsimulated configuration of antennal lobe

inp = 0.15 #input current amplitude
noise_amp = 0.0 #max noise percentage of inp
noise_test = 0.0 #no noise in the mixtures

num_odors_train = 3 #how many buckets. Base odors.
num_odors_mix = 2 #mix 2 of the odors together

num_alpha = 20 #values of A in: A*I_1 + (1-A)*I_2

num_test = 1 #test per value of A. If there is no noise, then this value may make no difference if greater than or equal
#   to one. If there is noise, then it may be useful to raise this above 1.


run_time = 120*ms

usingMixtures2 = False
usingMixtures3 = True

I_arr = []
#create the base odors (num_odors_train) of them
for i in range(num_odors_train):
    I = ex.get_rand_I(N_AL, p = 0.33, I_inp = inp)*nA
    I_arr.append(I)

run_params_train = dict(num_odors = num_odors_train,
                        num_trials = 1,
                        prefix = tr_prefix,
                        inp = inp,
                        noise_amp = noise_amp,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = True)


states = dict(  G_AL = G_AL,
                S_AL = S_AL,
                trace_AL = trace_AL,
                spikes_AL = spikes_AL)
 

run_params_test = dict( num_odors = num_odors_mix,
                        num_trials = num_alpha,
                        prefix = te_prefix,
                        inp = inp,
                        noise_amp = noise_test,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = False)

# --------------------------------------------------------------
# run the simulation and save to disk
ex.createData(run_params_train, I_arr, states, net)
if(usingMixtures2):
    ex.mixtures2(run_params_test, I_arr[:num_odors_mix], states, net)
if (usingMixtures3):
    ex.mixtures3(run_params_test, I_arr[:3], states, net)

#load in the data from disk
spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors_train)
#Counter
number_Of_Mix_Runs = 0 # necessary to increment like this because it is not known without first dividing the grid up
listOfAlpha = []
listOfBeta = []
for i in range(num_alpha):
    for j in range(num_alpha):
        A_arr = np.linspace(0, 1, num_alpha)
        B_arr = np.linspace(0, 1, num_alpha)
        if A_arr[i] + B_arr[j] <= 1:  # Only using normalized probabilities
            listOfAlpha.append(A_arr[i])
            listOfBeta.append(B_arr[j])
            number_Of_Mix_Runs+=1
if(usingMixtures2):
    spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_alpha*num_test)
if(usingMixtures3):
    spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = number_Of_Mix_Runs)

# dim(trace_V_arr) = (3, 1000, 2300) which is (num base odors, num neurons in network, # timesteps after start=100ms)
# dim(X) =  (6900, 1000) which is (num base odors*# timesteps after start=100ms, num neurons in network)
X = np.hstack(trace_V_arr).T

#normalize training (base odor) data
mini = np.min(X) # lowest voltage of any neuron's voltages at any time for any odor
maxi = np.max(X) # highest voltage of any neuron's voltages at any time for any odor
normalized_Training_Voltages = anal.normalize(X, mini, maxi) # 0 if minimum value, 1 if maximum value.

point_Labels = np.hstack(label_arr)

#train the SVM on base odors and labels
clf = anal.learnSVM(normalized_Training_Voltages, point_Labels)

test_data = test_V_arr # mixed odor voltage
test_data = anal.normalize(test_data, mini, maxi) # normalize test voltages using training mins and maxes

y_test = np.mean(label_test_arr, axis = 1)

pred_arr = []
A_arr = [] # will have dim = len(test_data) = 210
print("LENGTH of test_data: "+str(np.shape(test_data)))
B_arr = []
# The following comment numbers follow from num_alpha=20 with three base odors and two parameters (alpha and beta)
for i in range(len(test_data)): # for i from 0 to 54 inclusive, that is for all mixed odors individually
    pred = clf.predict(test_data[i].T) # attempt to classify mixed odors; dim=(2300,); test_data has shape (210,1000,2300) where 210 is number of test (AKA mixed) odors
    # so for this mixed odor i, clf.predict will predict for each time point what base odor it belongs to.
    total_pred = stats.mode(pred)[0] # dim=(1,); stats.mode(pred) has shape (2,1); index 0 returns mode instead of number of counts of mode.
    pred_arr.append(total_pred) # dim=(210,1) after loop is done, where 210 is the number of alpha beta grid points in plot.
    # One base odor classification mode per test mixture

    # histogram has dimension (num_odors_train,)=(3,)
    # dim=(210,3) after loop is done, where 210 is the number of alpha beta grid points in plot.
    A_arr.append(np.histogram(pred, bins = np.arange(num_odors_train+1))[0])
    # Adds histogram (which is the 0th component) of object for each mix
    # This is dimension (mixed, base). It shows how much of the mixed is identified as each base
print("LENGTH of pred: "+str(np.shape(pred)))
print("LENGTH of total_pred: "+str(np.shape(total_pred)))
print("LENGTH of pred_arr: "+str(np.shape(pred_arr)))
print("LENGTH of A_arr: "+str(np.shape(A_arr)))

A_arr = np.array(A_arr)/np.sum(A_arr[0]) #alpha array; dim (210,3); from 0 to 1, the proportion of
# each of the 2300 dots in time series for each specific mixture (row) identified as the base odor (column)
# Normalized by mixed odor 1 sum because it doesn't matter which one we choose (adds to 2300, the number of time points)

np.savetxt(tr_prefix+'alpha_histogram.txt', A_arr, fmt = '%1.3f')
expected = y_test
predicted = np.array(pred_arr)

odor1Proportion = A_arr[:, 0] # dim 210; for all mixtures, proportion in base 1
odor2Proportion = A_arr[:, 1] # dim 210; for all mixtures, proportion in base 2
odor3Proportion = A_arr[:, 2] # dim 210; for all mixtures, proportion in base 3

# Checking normalization of sum of different class predictions
# for i in range(0,55):
#     print(x[i]+y[i]+z[i])

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

A = np.arange(num_alpha)/num_alpha


#----------------------------------------------------
#Plotting

if(usingMixtures2):
    popt_x, pcov_x = curve_fit(fsigmoid, A, odor1Proportion)
    popt_y, pcov_y = curve_fit(fsigmoid, A, odor2Proportion)

    # print(popt_x)
    # print(popt_y)
    plt.plot(A, odor1Proportion, 'r.', label = r'$P(I_1)$')
    plt.plot(A, fsigmoid(A, *popt_y), 'r-', linewidth = 2)
    plt.plot(A, odor2Proportion, 'b.', label = r'$P(I_2)$')
    plt.plot(A, fsigmoid(A, *popt_x), 'b-', linewidth = 2)
    plt.plot(A, odor3Proportion, 'k.', label = r'$P(I_3)$')
    plt.title('Classification of the Mixture of 3 Odors', fontsize = 20)
    plt.xlabel(r'$\alpha$ in $\alpha I_1 + (1-\alpha) I_2$', fontsize = 16)
    plt.ylabel(r'Classification Probability of $I_1/I_2$', fontsize = 16)


if(usingMixtures3):
    fig=plt.figure()
    figSubplot = fig.add_subplot(111, projection='3d')
    figSubplot.scatter(listOfAlpha,listOfBeta,odor1Proportion, color="r", label="Amount in odor 1")
    figSubplot.scatter(listOfAlpha,listOfBeta,odor2Proportion, color="b", label="Amount in odor 2")
    figSubplot.scatter(listOfAlpha,listOfBeta,odor3Proportion, color="y", label="Amount in odor 3")
    figSubplot.set_xlabel('Alpha')
    figSubplot.set_ylabel('Beta')

plt.legend()

# Plotting best fit meshes
# Softmax attempt
def softMaxProbability(A_val, B_val, a, b, c):
    C_val = 1.0-A_val-B_val
    A_term = np.exp(a*np.power(A_val,1))
    B_term = np.exp(b*np.power(B_val,1))
    C_term = np.exp(c*np.power(C_val,1))
    # The softmax denominator is named Z
    Z = A_term + B_term + C_term
    return np.array([np.divide(A_term,Z), np.divide(B_term,Z), np.divide(C_term,Z)])

# For plotting: Full alpha and beta range (square)
xAxisArray = 1-np.outer(np.linspace(1,0,2*num_alpha), np.ones(2*num_alpha))
yAxisArray = xAxisArray.copy().T

# Need one surface fit for each base odor probability distribution over alpha and beta
# These parameters are set by hand but can be tuned by a computer algorithm if necessary.
a = 20
b = 20
c = 20
# Calculate softmax fit at each parameter point (alpha, beta)
FitOdor1Array, FitOdor2Array, FitOdor3Array = softMaxProbability(xAxisArray, yAxisArray, a,b,c)
# each ProbOdor array has dim = (40,40)

# Remove prediction for (alpha, beta) not in valid range [0,1] so that unnecessary mesh is not plotted
for row in range(len(xAxisArray)):
    for col in range(len(xAxisArray[0])):
        if (xAxisArray[row][col] + yAxisArray[row][col]> 1.0):
            FitOdor1Array[row][col] = NaN
            FitOdor2Array[row][col] = NaN
            FitOdor3Array[row][col] = NaN

# fig = plt.figure()
# ax = plt.axes(projection='3d')
figSubplot.plot_surface(xAxisArray, yAxisArray, FitOdor1Array,color='r', edgecolor='none', alpha = 0.3)
figSubplot.plot_surface(xAxisArray, yAxisArray, FitOdor2Array,color='b', edgecolor='none', alpha = 0.3)
figSubplot.plot_surface(xAxisArray, yAxisArray, FitOdor3Array,color='y', edgecolor='none', alpha = 0.3)

plt.show()

plt.savefig('mixtures.pdf', bbox = 'tight')


