# ---- Important Functions --- #
import pandas as pd

import torchvision.transforms as T
import torch
import numpy as np

from scipy import stats
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms#
from IPython.core.pylabtools import figsize
import itertools
import scipy.stats as st

from IPython.core.pylabtools import figsize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Function to obtain a sample of probabilities --- #
# This function runs the model with dropout layers set to "training"
# for the number of samples the user wishes to obtain
def sampler(model, inputs, num_samples):
    outputs = []
    model.eval() # sets all layers to eval
    model.dropout.train() # resets dropout to train
    for i in range(num_samples):
        output = model(inputs) #Run the model for how many samples you want
        
        outputs.append(output.detach().cpu())
    return outputs

# --- Function to format the probabilities --- #
# This function formats probabilities outputted by the sampler function into
# a format to be used for further analysis
def form_probs(probs, num_samples):
    prob_form = []
    for i in range(len(probs[0])):
        prob_ind = []
        for j in range(num_samples):
            prob_ind.append(probs[j][i].cpu().detach().numpy())
        prob_ind = np.stack(prob_ind,axis = 0)#.reshape((num_samples,2))
        prob_form.append(prob_ind)
    return prob_form

# --- Function that produces performance measures --- #
# This function produces redults for all the performance measures
def accuracy_measures(net,X, n_samples = 1000, cal_val1 = 0, cal_val2 = 1):
  
  # Set up all the empty arrays of the performance measures needed
  mean_pred = np.empty(len(X))
  mode_pred = np.empty(len(X))
  median_pred = np.empty(len(X))
  dist_pred = np.empty(len(X))
  cred_pred = [False]  * len(X)
  cred_centered_pred = [False]  * len(X)
  cred_oneside_pred = [False]  * len(X)
  count = 0

  bins = np.linspace(0, 1, num=101)
  dist_pred = [0]  * len(X)
  shannon = []
  shannon_correct = []
  shannon_incorrect = []
  true_value_cal_data = []
  pred_value_cal_data = []
  y = []
  
  for i in range(len(X)):
    image, label, name, agreement_level = X[i]
    
    # Get a unique image
    image = image[None, :, :]
    y.append(label)
    # obtain a sample of probabilities from the neural network for the given image
    probs_outputs = sampler(net, image.to(device), n_samples)
    probs = form_probs(probs_outputs, n_samples)
    # Calculate mean probability
    mean_prob = np.mean(probs)
    # Calculate the output Y based on mean probability
    mean_pred[i] = (np.array(mean_prob) > 0.5)
    # Calculate the median probability
    median_prob = np.median(probs)
    #Calculate the output Y based on median probability
    median_pred[i] = (np.array(median_prob) > 0.5)
    # Calculate the mode probability
    mode_prob = stats.mode(np.around(np.array(probs).flatten(),2))[0]
    #Calculate the output Y based on mode probability
    mode_pred[i] = (np.array(mode_prob) > 0.5)
    
    # Calculate the 95% Credible interval of the sample of probabilities
    percentile_2_5 = np.percentile(probs, 2.5)
    percentile_97_5 = np.percentile(probs, 97.5)
    # Calculate if the credible interval contains allows for the true observation Y
    if (0.5 >= percentile_2_5) & (0.5 <= percentile_97_5):
      cred_pred[i] = True
      cred_centered_pred[i] = True
      count += 1
    if (0.5 < percentile_2_5) & (label == 1):
      cred_pred[i] = True
      cred_oneside_pred[i] = True
      count += 1
    if (0.5 > percentile_97_5) & (label == 0):
      cred_pred[i] = True
      cred_oneside_pred[i] = True
      count += 1
    
    # Calculate which side has the greatest area under the curve for the probabilities histogram
    N, bin_borders = np.histogram(probs, bins = bins,density=True)
    ind_05 = np.where(bins==0.5)[0]
    sum_greater_05 = np.sum(np.array(N)[int(ind_05):]*0.01)
    sum_less_05 = np.sum(N[:int(ind_05)]*0.01)
    if sum_greater_05 > sum_less_05:
      dist_pred[i] = 1

    # Calculate the entropy of each observation
    bin_widths = np.diff(bin_borders)
    px1 = N* bin_widths
    log_ent =1/np.array(px1)[np.nonzero(px1)]
    sumpx = sum(np.array(px1)[np.nonzero(px1)]* np.log(log_ent))
    if cal_val1 <= entropy(entropy(px1, base =len(px1))) <= cal_val2:
      true_value_cal_data.append(label)
      pred_value_cal_data.append(mean_prob)
    shannon.append(entropy(px1, base =len(px1)))
    # Assign labels to assess whether entropy aligns with a correct prediction 
    # or an incorrect prediction
    if cred_pred[i] == True:
      shannon_correct.append(entropy(px1, base =len(px1)))
    if cred_pred[i] != True:
      shannon_incorrect.append(entropy(px1, base =len(px1)))
  
  #Calculate the overal accuracy measures for all observations
  mean_acc = (mean_pred == y).sum()/len(X)
  mode_acc = (mode_pred == y).sum()/len(X)
  median_acc = (median_pred == y).sum()/len(X)
  cred_acc = np.array(cred_pred).sum()/len(X)
  cred_centered_acc = np.array(cred_centered_pred).sum()/len(X)
  cred_oneside_acc = np.array(cred_oneside_pred).sum()/len(X)
  dist_acc = (np.array(dist_pred) == np.array(y)).sum()/len(X)
  # Print the performance measures statistics
  print("Accuracy using mean probability: ", round(mean_acc*100,2), "%")
  print("Accuracy using mode probability: ", round(mode_acc*100,2), "%")
  print("Accuracy using median probability: ", round(median_acc*100,2), "%")
  print("Accuracy using 95% Credible Interval: ", round(cred_acc*100,2), "%")
  print("Percentage of the 95% Credible Interval Correct predictions that are centered: ", round((round(cred_centered_acc*100,2)/round(cred_acc*100,2))*100,2), "% (Only looking at those CIs which contain 0.5)")
  print("Percentage of the 95% Credible Interval Correct predictions that are either side of 0.5: ", round((round(cred_oneside_acc*100,2)/round(cred_acc*100,2))*100,2), "% (Only looking at those CIs where the 95% is below or above 0.5)")
  print("Accuracy using the greatest area under the curve on either side of 0.5: ", round(dist_acc*100,2), "%")

  
  return mean_acc, mode_acc, median_acc, cred_acc, cred_centered_acc, cred_oneside_acc, dist_acc, shannon, shannon_correct, shannon_incorrect, true_value_cal_data, pred_value_cal_data, mean_pred,y

# --- Function to calculate the F1 Score --- #
def calc_metrics(predictions, y_test):
    accuracy = np.mean(predictions == y_test)
    f1_metric = f1_score(y_test, predictions)

    print('Accuracy of Model: {:.2f}%'.format(100 * accuracy))
    print('F1 Score of Model: {:.4f}'.format(f1_metric)) 
    return f1_metric 

# --- Function to plot the confusion matrix --- #
def plot_confusion_matrix(k,cm, classes = ['Accept', 'Reject'],
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    
    # Display the matrix in text form
    print('Confusion matrix')
    print(cm)
    figsize(8, 8)
    
    # Show the matrix using the imshow functionality
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 20)
    
    # Tick marks show classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 12)
    plt.yticks(tick_marks, classes, rotation = 90, size = 12)

    # Formatting for text labels on plot
    fmt1 = 's'
    fmt2 = 'f'
    thresh = cm.max() / 2.
    
    # Four types of classifications
    types = [['True Negative', 'False Positive'],
             ['False Negative', 'True Positive']]
    
    # Add the actual numbers and the types onto the heatmap plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i - 0.05, format(types[i][j], fmt1),
                 horizontalalignment="center", size = 18,
                 color="white" if cm[i, j] > thresh else "black")
        percent = round(cm[i, j]/sum(sum(cm))*100,2)
        plt.text(j, i + 0.15, str(percent) + '%',
                 horizontalalignment="center", size = 24,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size = 16)
    plt.xlabel('Predicted Label', size = 16)

    #User to change the filename
    plt.savefig("Confusion Matrix File"+str(k)+'.png')  
    #plt.show()

# --- Main function for performance measures for the test set --- #
# This function is the main function to get all performance measures
# for a test set after training is completed
# All performance measures were written to files 

def investigate_obs2(probs, obs, true_label, image_name,examine_pred = True):

    # Creat file for all the data to be saved to.   
    file_name = "Test Set Performance measures file"+str(image_name)+'.txt'
    with open(file_name, 'w') as ff:
      probs = np.array(pd.DataFrame(probs)).flatten() #Probs Diagnosed
      #Fit the conjugate prior - a beta distribution
      best_dist = "Beta"
      params = st.beta.fit(probs,method="MM",floc = 0, fscale = 1)
      dist_str = "The fitted probability distribution is a "+str(best_dist)+ " distribution with parameters: "+str(params) 
      print(dist_str)
      ff.write(dist_str)
      figsize(6, 6)
      x2 = np.linspace(min(probs), max(probs), 100)
      y2 = st.beta.pdf(x2, params[0], params[1], params[2], params[3])
      
    # Print the information about the prediction
      median_prob =np.median(probs)
      tag = 'HP'
      if (np.array(median_prob) > 0.5):
        tag = "SSA"
      print('\n\nMedian estimated probability: {:0.3f}'.format(median_prob), " Median Prediction: ", tag)
      ff.write("\n")
      ff.write('Median estimated probability: {:0.3f}'.format(median_prob))
      ff.write("\n")
      ff.write(f"Median Prediction: {tag}")
     
    
    #Mean
      mean_prob = np.mean(probs)
      tag = 'HP'
      if (np.array(mean_prob) > 0.5):
        tag = 'SSA'
      print('\n\nMean estimated probability: {:0.3f}'.format(mean_prob), " Mean Prediction: ", tag)
      ff.write("\n")
      ff.write('Mean estimated probability: {:0.3f}'.format(mean_prob))
      ff.write("\n")
      ff.write(f"Mean Prediction: {tag}")
    #Mode
      mode_prob = stats.mode(np.around(np.array(probs).flatten(),2))[0]
      tag = 'HP'
      if (np.array(mode_prob) > 0.5):
        tag = "SSA"
      print('\n\nMode estimated probability: ',mode_prob.item(), " Mean Prediction: ", tag)
      ff.write("\n")
      ff.write('Mode estimated probability: {:0.3f}'.format(mode_prob.item()))
      ff.write("\n")
      ff.write(f"Mode Prediction: {tag}")
    #Majority


      print('2.5% estimated probability:     {:0.3f}'.format(np.percentile(probs, 2.5)))
      print('97.5% estimated probability:    {:0.3f}\n\n'.format(np.percentile(probs, 97.5)))
      ff.write("\n")
      ff.write('2.5% estimated probability:     {:0.3f}'.format(np.percentile(probs, 2.5)))
      ff.write("\n")
      ff.write('97.5% estimated probability:    {:0.3f}\n\n'.format(np.percentile(probs, 97.5)))
      
      bins = np.linspace(0, 1, num=101)
      N, bin_borders = np.histogram(probs, bins = bins,density=True)
    
      bin_widths = np.diff(bin_borders)
      px1 = N* bin_widths
      shannon = entropy(px1, base =len(px1))

      print('Entropy:    {:0.3f}\n\n'.format(shannon))
      ff.write("\n")
      ff.write('Entropy:    {:0.3f}\n\n'.format(shannon))
      ff.write("\n")
    
    # Density Plot of the Probabilities
    plt.figure()
    hist = sns.histplot(np.array(probs).flatten(), label = 'Samples', stat = 'density')#, bins = 20)
    plt.xlim(0,1)
    # Vertical lines at ranges of the credible interval
    plt.vlines([np.percentile(probs, 2.5), np.percentile(probs, 97.5)], 
               ymin = 0, ymax = hist.get_ybound()[1],  colors = ['red'], linestyles = '--',
              label = '2.5% and 97.5% CI');
    plt.plot(x2,y2, color = "orange", lw=2, ls='-', alpha=0.5, label='Fitted Dist.');
    # Plot labels
    plt.xlabel('P(Getting Colon Cancer)', size = 18); plt.ylabel('Density', size =18);
    
    plt.title('True Class: %s' % ('SSA' if true_label == 1 else 'HP'), size = 18);
    plt.legend(prop={'size': 12});
    plt.savefig("ElicitedUncertaintyDistribution"+str(image_name)+'.png')  
    plt.figure()
    #plt.show()