# --- Deep Learning Model --- #
import pandas as pd

import torchvision.transforms as T
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from scipy import stats
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.calibration import calibration_curve
#matplotlib inline

import matplotlib.lines as mlines


from KeyFunctions import investigate_obs2,plot_confusion_matrix,calc_metrics,accuracy_measures,form_probs,sampler
from ResnetModel import ResNet_18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- Train model function --- #

def train_pytorch_cnn_dropout(j, dataset, batch_size, epochs, num_samples):
  #split dataset into training and validation
  train_set, val_set = torch.utils.data.random_split(dataset, [int(.8*len(dataset)),int(.2*len(dataset)+1)], generator=torch.Generator().manual_seed(j))
  

  train_dataloader =  DataLoader(train_set, batch_size=batch_size,
                        shuffle=True)
  val_dataloader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=True)
  #Un comment this section if you want to add an earlier stopper
  #early_stopper = EarlyStopper(patience=3, min_delta=0.1)

  #load model
  net = ResNet_18(3, 1).to(device)
  
  criterion = nn.BCELoss()
  optimizer = optim.SGD(net.parameters(), lr= 1e-3, momentum=0.9)
  #scheduler to change the LR for each epoch
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.01)
  
  file_name = "TrainingOutput"+str(j)+'.txt'
  with open(file_name, 'w') as ff:
    
  #Training
    for epoch in range(epochs):  # loop over the dataset multiple times
      correct = 0
      correct_mean = 0
      total = 0 
      running_loss = 0.0
      running_loss_v = 0.0
      for i, data in enumerate(train_dataloader,0):
        # Set network to training
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        # get a sample of outputs and format them
        probs_outputs = sampler(net, inputs, num_samples)
        formatted_probs = form_probs(probs_outputs, num_samples)
        
        running_loss += loss.item()
        # get and print some accuracy measures to track progress
        predicted = np.array([int(b) for b in (outputs>0.5)])
        mean_probs = np.array([np.mean(b) for b in formatted_probs])
        predicted_mean = np.array([int(b) for b in (mean_probs>0.5)])
        total += labels.size(0)
        correct += (predicted == labels.cpu().numpy()).sum().item()
        correct_mean += (predicted_mean == labels.cpu().numpy()).sum().item()

      print(f'[{epoch + 1}/ {epochs}] loss: {running_loss / len(train_set)}')
      print("\n")
      print(f'Accuracy of the network: {100 * correct // total} %')
      print("\n")
      print(f'Mean Accuracy of the network: {100 * correct_mean // total} %')
      print("\n")  
      ff.write(f'[{epoch + 1}/ {epochs}] loss: {running_loss / len(train_set)}')
      ff.write("\n")
      ff.write(f'Accuracy of the network: {100 * correct // total} %')
      ff.write("\n")
      ff.write(f'Mean Accuracy of the network: {100 * correct_mean // total} %')
      ff.write("\n")
      correct = 0
      correct_mean = 0
      total = 0 
      count_ones = 0
      # Take the network out of training to run validation on the current state
      with torch.no_grad():
        for i, data in enumerate(val_dataloader,0):
          images, labels = data
          images = images.to(device)
          labels = labels.to(device)
          # calculate outputs by running images through the network
          outputs = net(images)
          probs_outputs = sampler(net, images, num_samples)
          formatted_probs = form_probs(probs_outputs, num_samples)
          mean_probs = np.array([np.mean(b) for b in formatted_probs])
          predicted_mean = np.array([int(b) for b in (mean_probs>0.5)])
          count_ones += labels.sum().item()
          loss = criterion(outputs.float(), labels.float())
          running_loss_v += loss.item()
        
          predicted = np.array([int(b) for b in (outputs>0.5)])
          total += labels.size(0)
          correct += (predicted == labels.cpu().numpy()).sum().item()
          correct_mean += (predicted_mean == labels.cpu().numpy()).sum().item()
        print(f'Percentage of Ones in Validation set: {100 * count_ones // total} %') 
        print("\n")   
        print(f'Validation loss: {running_loss_v / len(val_set)}') 
        print("\n")
        print(f'Accuracy of the Validation set: {100 * correct // total} %')
        print("\n")
        print(f'Mean Accuracy of the Validation set: {100 * correct_mean // total} %')
        ff.write("\n")
        ff.write(f'Percentage of Ones in Validation set: {100 * count_ones // total} %') 
        ff.write("\n")   
        ff.write(f'Validation loss: {running_loss_v / len(val_set)}') 
        ff.write("\n")
        ff.write(f'Accuracy of the Validation set: {100 * correct // total} %')
        ff.write("\n")
        ff.write(f'Mean Accuracy of the Validation set: {100 * correct_mean // total} %')
        ff.write("\n")
        # -- Un comment this code if you want to implement early stopping
        #if early_stopper.early_stop(running_loss_v):             
          #break
      # After ten epochs start changing the learning ratio    
      if epoch > 10:
        scheduler.step()
    ff.write('Finished Training')
    ff.write("\n")
    ff.write("\n")
    ff.write("\n")
    print('Finished Training')
    # once training is completed we can observe all the accuracy measure for the testing set
  # Testing 
    mean_acc, mode_acc, median_acc, cred_acc, cred_centered_acc, cred_oneside_acc, dist_acc, shannon, shannon_correct, shannon_incorrect, cal_values_true, cal_values_pred , mean_pred, labels_test= accuracy_measures(net, val_set,  n_samples = 500, cal_val1 = 0, cal_val2 = 1)
    ff.write(f'Accuracy using mean probability: {round(mean_acc*100,2)} %')
    ff.write("\n")
    ff.write(f'Accuracy using mode probability:  {round(mode_acc*100,2)} %')
    ff.write("\n")
    ff.write(f'Accuracy using median probability: { round(median_acc*100,2)} %')
    ff.write("\n")
    ff.write(f'Accuracy using 95% Credible Interval:  {round(cred_acc*100,2)} %')
    ff.write("\n")
    ff.write(f'Percentage of the 95% Credible Interval Correct predictions that are centered:  {round((round(cred_centered_acc*100,2)/round(cred_acc*100,2))*100,2)} % (Only looking at those CIs which contain 0.5)')
    ff.write("\n")
    ff.write(f'Percentage of the 95% Credible Interval Correct predictions that are either side of 0.5:  {round((round(cred_oneside_acc*100,2)/round(cred_acc*100,2))*100,2)} % (Only looking at those CIs where the 95% is below or above 0.5)')
    ff.write("\n")
    ff.write(f'Accuracy using the greatest area under the curve on either side of 0.5:  {round(dist_acc*100,2)} %')
    ff.write("\n")
    f_value = calc_metrics(mean_pred, labels_test)
    ff.write(f'F1 Score: {f_value}')
    ff.write("\n")

  # Get plots for performance
  plt.figure()
  shannon_plot = sns.histplot(shannon, label = 'pdf', stat = 'proportion')
  plt.xlabel('Entropy', size = 18); plt.ylabel('Proportion', size =18);
  plt.savefig("AllEntropyPlot"+str(j)+'.png')    
  plt.figure()

  shannon_plot_correct = sns.histplot(shannon_correct, label = 'pdf', stat = 'proportion')

  plt.xlabel('Entropy of correct predictions (based on CI)', size = 18); plt.ylabel('Proportion', size =18);
  plt.savefig("CorrectEntropyPlot"+str(j)+'.png')  
  plt.figure()

  shannon_plot_incorrect = sns.histplot(shannon_incorrect, label = 'pdf', stat = 'proportion')

  plt.xlabel('Entropy incorrect predictions (based on CI)', size = 18); plt.ylabel('Proportion', size =18);
  plt.savefig("IncorrectEntropyPlot"+str(j)+'.png')    
  plt.figure()

  #Calibration Plot
  cal_y, cal_x = calibration_curve(np.array(cal_values_true), cal_values_pred, n_bins = 10)

  fig, ax = plt.subplots()
  # only these two lines are calibration curves
  plt.plot(cal_x,cal_y, marker='o', linewidth=1, label='reg')


  # reference line, legends, and axis labels
  line = mlines.Line2D([0, 1], [0, 1], color='black')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()
  plt.savefig("CalibrationPlot"+str(j)+'.png')  
  plt.figure()

  #Confusion Matrix
  # Threshold the values at 0.5
  predictions = (np.array(mean_pred) > 0.5)
#print(predictions, y_test)
  print(calc_metrics(mean_pred, labels_test))

  cm = confusion_matrix(labels_test, mean_pred)
  plot_confusion_matrix(j,cm, classes = ['HP', 'SSA'])

  #Return the training network and the testing set for any further analysis
  return net,val_set

## ---- Function to elicit distributions for new observations --- #

def elicit_distribution(net,dataset_testing):
  
  for j in range(len(dataset_testing)):
    data = dataset_testing[j]
    image = data[0]
    label = data[1]
    image_name = data[2]
    image = image.to(device)
    image = image[None, :, :, :]
    
          # calculate outputs by running images through the network
    probs_outputs = sampler(net, image, 100)
    #print(probs_outputs)
    formatted_probs = form_probs(probs_outputs, 100)
    #print(formatted_probs)
    investigate_obs2(formatted_probs, image, label, image_name,examine_pred = True)

## ---- Function for model diagnostics with inter-rater agreement --- #
def agreement_stats(dataset_testing,net):
  zero_agree = []
  count_CI_zero = 0
  one_agree = []
  count_CI_one = 0
  two_agree = []
  count_CI_two = 0
  three_agree = []
  count_CI_three = 0
  four_agree = []
  count_CI_four = 0
  five_agree = []
  count_CI_five = 0
  six_agree = []
  count_CI_six = 0
  seven_agree = []
  count_CI_seven = 0
  
  ent_zero_agree = []
  ent_one_agree = []
  ent_two_agree = []
  ent_three_agree = []
  ent_four_agree = []
  ent_five_agree = []
  ent_six_agree = []
  ent_seven_agree = []
  
  meanent_zero_agree = []
  meanent_one_agree = []
  meanent_two_agree = []
  meanent_three_agree = []
  meanent_four_agree = []
  meanent_five_agree = []
  meanent_six_agree = []
  meanent_seven_agree = []
  
  modeent_zero_agree = []
  modeent_one_agree = []
  modeent_two_agree = []
  modeent_three_agree = []
  modeent_four_agree = []
  modeent_five_agree = []
  modeent_six_agree = []
  modeent_seven_agree = []
  
  medianent_zero_agree = []
  medianent_one_agree = []
  medianent_two_agree = []
  medianent_three_agree = []
  medianent_four_agree = []
  medianent_five_agree = []
  medianent_six_agree = []
  medianent_seven_agree = []
  
  for j in range(len(dataset_testing)):
    data = dataset_testing[j]
    image = data[0]
    label = data[1]
    image_name = data[2]
    agreement = data[3]
    image = image.to(device)
    image = image[None, :, :, :]
    
          # calculate outputs by running images through the network
    probs_outputs = sampler(net, image, 100)
    formatted_probs = form_probs(probs_outputs, 100)
    probs = np.array(pd.DataFrame(formatted_probs)).flatten()
    bins = np.linspace(0, 1, num=101)
    N, bin_borders = np.histogram(probs, bins = bins,density=True)
    
    bin_widths = np.diff(bin_borders)
    px1 = N* bin_widths
    shannon = entropy(px1, base =len(px1))
    lower_quant = np.percentile(probs, 2.5)
    upper_quant = np.percentile(probs, 97.5)
    mean_probs = np.mean(probs)
    mode_prob = stats.mode(np.around(np.array(probs).flatten(),2))[0]
    median_prob = np.median(probs)
      
    if agreement == 0:
      ent_zero_agree.append(shannon)
      zero_agree.append(probs)
      
      meanent_zero_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_zero_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_zero_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_zero +=1
    if agreement == 1:
      ent_one_agree.append(shannon)
      one_agree.append(probs)
      
      meanent_one_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_one_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_one_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_one +=1
    if agreement == 2:
      ent_two_agree.append(shannon)
      two_agree.append(probs)
      
      meanent_two_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_two_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_two_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_two +=1
    if agreement == 3:
      ent_three_agree.append(shannon)
      three_agree.append(probs)
      
      meanent_three_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_three_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_three_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_three +=1
    if agreement == 4:
      ent_four_agree.append(shannon)
      four_agree.append(probs)
      
      meanent_four_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_four_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_four_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_four +=1
    if agreement == 5:
      ent_five_agree.append(shannon)
      five_agree.append(probs)
      
      meanent_five_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_five_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_five_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_five +=1
    if agreement == 6:
      ent_six_agree.append(shannon)
      six_agree.append(probs)
      
      meanent_six_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_six_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_six_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant> 0.5 :
        count_CI_six +=1
    if agreement == 7:
      ent_seven_agree.append(shannon)
      seven_agree.append(probs)
      
      meanent_seven_agree.append(entropy(np.array([mean_probs,1-mean_probs]),base =2))
      modeent_seven_agree.append(entropy(np.array([mode_prob,1-mode_prob]),base =2))
      medianent_seven_agree.append(entropy(np.array([median_prob,1-median_prob]),base =2))
      if lower_quant < 0.5 and upper_quant > 0.5 :
        count_CI_seven +=1
        
  return zero_agree, count_CI_zero,one_agree,count_CI_one,two_agree,count_CI_two,three_agree,count_CI_three,four_agree,count_CI_four,five_agree ,count_CI_five ,six_agree ,count_CI_six,seven_agree ,count_CI_seven,ent_zero_agree,ent_one_agree,ent_two_agree,ent_three_agree,ent_four_agree,ent_five_agree ,ent_six_agree ,ent_seven_agree,meanent_zero_agree,meanent_one_agree,meanent_two_agree,meanent_three_agree,meanent_four_agree,meanent_five_agree,meanent_six_agree,meanent_seven_agree,modeent_zero_agree,modeent_one_agree,modeent_two_agree,modeent_three_agree,modeent_four_agree,modeent_five_agree,modeent_six_agree,modeent_seven_agree,medianent_zero_agree,medianent_one_agree,medianent_two_agree,medianent_three_agree,medianent_four_agree,medianent_five_agree,medianent_six_agree,medianent_seven_agree

# --- Function which reports model diagnostics for inter-rater agreement --- #      
def agreementreporting(testing_set,net):
  zero_agree, count_CI_zero,one_agree,count_CI_one,two_agree,count_CI_two,three_agree,count_CI_three,four_agree,count_CI_four,five_agree ,count_CI_five ,six_agree ,count_CI_six,seven_agree ,count_CI_seven,ent_zero_agree,ent_one_agree,ent_two_agree,ent_three_agree,ent_four_agree,ent_five_agree ,ent_six_agree ,ent_seven_agree,meanent_zero_agree,meanent_one_agree,meanent_two_agree,meanent_three_agree,meanent_four_agree,meanent_five_agree,meanent_six_agree,meanent_seven_agree,modeent_zero_agree,modeent_one_agree,modeent_two_agree,modeent_three_agree,modeent_four_agree,modeent_five_agree,modeent_six_agree,modeent_seven_agree,medianent_zero_agree,medianent_one_agree,medianent_two_agree,medianent_three_agree,medianent_four_agree,medianent_five_agree,medianent_six_agree,medianent_seven_agree = agreement_stats(testing_set,net)
  file_name = 'AgreementReporting.txt'
  with open(file_name, 'w') as ff: 
    ff.write('Agreement Levels Reportings')
    ff.write("\n")
    ff.write('Distribution Entropy and Credible Intervals')
    ff.write("\n")
    ff.write('Agreement Level Zero')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_zero_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_zero_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_zero_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_zero/len(ent_zero_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_zero}')
    ff.write("\n")
    ff.write(f'Count of Zero:  {len(ent_zero_agree)}')
    ff.write("\n")
    ff.write('Agreement Level One')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_one_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_one_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_one_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_one/len(ent_one_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_one}')
    ff.write("\n")
    ff.write(f'Count of One:  {len(ent_one_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Two')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_two_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_two_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_two_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_two/len(ent_two_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_two}')
    ff.write("\n")
    ff.write(f'Count of Two:  {len(ent_two_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Three')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_three_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_three_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_three_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_three/len(ent_three_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_three}')
    ff.write("\n")
    ff.write(f'Count of Three:  {len(ent_three_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Four')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_four_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_four_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_four_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_four/len(ent_four_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_four}')
    ff.write("\n")
    ff.write(f'Count of Four:  {len(ent_four_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Five')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_five_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_five_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_five_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_five/len(ent_five_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_five}')
    ff.write("\n")
    ff.write(f'Count of Five:  {len(ent_five_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Six')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_six_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_six_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_six_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_six/len(ent_six_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_six}')
    ff.write("\n")
    ff.write(f'Count of Six:  {len(ent_six_agree)}')
    ff.write("\n")
    ff.write('Agreement Level Seven')
    ff.write("\n")
    ff.write(f'Mean Entropy:  {np.mean(ent_seven_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Entropy:  {np.median(ent_seven_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Entropy:  {stats.mode(np.around(np.array(ent_seven_agree).flatten(),2))[0]} ')
    ff.write("\n")
    frac_nCI = count_CI_seven/len(ent_seven_agree)
    ff.write(f'Percentage of Centered CI:  {round(frac_nCI * 100,2)} %')
    ff.write("\n")
    ff.write(f'Count of Centered CI:  {count_CI_seven}')
    ff.write("\n")
    ff.write(f'Count of Seven:  {len(ent_seven_agree)}')
    ff.write("\n")
    
    ff.write('Point Estimate Entropy')
    ff.write("\n")
    ff.write('Agreement Level Zero')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_zero_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_zero_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_zero_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_zero_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_zero_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_zero_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_zero_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_zero_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_zero_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level One')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_one_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_one_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_one_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_one_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_one_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_one_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_one_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_one_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_one_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Two')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_two_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_two_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_two_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_two_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_two_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_two_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_two_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_two_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_two_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Three')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_three_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_three_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_three_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_three_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_three_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_three_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_three_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_three_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_three_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Four')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_four_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_four_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_four_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_four_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_four_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_four_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_four_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_four_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_four_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Five')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_five_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_five_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_five_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_five_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_five_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_five_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_five_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_five_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_five_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Six')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_six_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_six_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_six_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_six_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_six_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_six_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_six_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_six_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_six_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write('Agreement Level Seven')
    ff.write("\n")
    ff.write(f'Mean Mean Entropy:  {np.mean(meanent_seven_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mean Entropy:  {np.median(meanent_seven_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mean Entropy:  {stats.mode(np.around(np.array(meanent_seven_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Median Entropy:  {np.mean(medianent_seven_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Median Entropy:  {np.median(medianent_seven_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Median Entropy:  {stats.mode(np.around(np.array(medianent_seven_agree).flatten(),2))[0]} ')
    ff.write("\n")
    ff.write(f'Mean Mode Entropy:  {np.mean(modeent_seven_agree)} ')
    ff.write("\n")  
    ff.write(f'Median Mode Entropy:  {np.median(modeent_seven_agree)} ')
    ff.write("\n")
    ff.write(f'Mode Mode Entropy:  {stats.mode(np.around(np.array(modeent_seven_agree).flatten(),2))[0]} ')
    ff.write("\n")
  #Plot Distribution Entropy with all agreement levels
  plt.figure()
  ax = sns.violinplot(data=[np.array(ent_zero_agree),np.array(ent_one_agree),np.array(ent_two_agree),np.array(ent_three_agree),np.array(ent_four_agree),np.array(ent_five_agree),np.array(ent_six_agree),np.array(ent_seven_agree)], cut = 0)
  ax.set_title("Distribution Entropy of Test Points by Different Agreement Levels")    
  ax.set_xlabel("Number of Pathologist Diagnosis SSA")
  ax.set_ylabel("Entropy")
  plt.savefig('AgreementDistEntropyAlllevels.png')
  data_fa = ent_zero_agree + ent_seven_agree 
  data_1d = ent_one_agree + ent_six_agree
  data_2d = ent_two_agree + ent_five_agree
  data_3d = ent_three_agree + ent_four_agree
  plt.figure()
  ax = sns.violinplot(data=[np.array(data_fa, dtype=float),np.array(data_1d, dtype=float),np.array(data_2d, dtype=float),np.array(data_3d, dtype=float)], cut = 0)  
  ax.set_title("Distribution Entropy of Test Points by Different Agreement Levels")   
  ax.set_xlabel("Number of Opposing Pathologist Diagnosis")
  ax.set_ylabel("Entropy")
  plt.savefig('AgreementDistEntropyFourlevels.png')
  
  #Plot Point Estimate Entropy
  #Mean
  plt.figure()
  ax = sns.violinplot(data=[np.array(meanent_zero_agree),np.array(meanent_one_agree),np.array(meanent_two_agree),np.array(meanent_three_agree),np.array(meanent_four_agree),np.array(meanent_five_agree),np.array(meanent_six_agree),np.array(meanent_seven_agree)], cut = 0) 
  ax.set_title("Mean Point Estimate Entropy of Test Points by Different Agreement Levels")    
  ax.set_xlabel("Number of Pathologist Diagnosis SSA")
  ax.set_ylabel("Entropy")   
  plt.savefig('AgreementMeanPointEntropyAlllevels.png')
  data_meanfa = meanent_zero_agree+meanent_seven_agree
  data_mean1d = meanent_one_agree+meanent_six_agree
  data_mean2d = meanent_two_agree+meanent_five_agree
  data_mean3d = meanent_three_agree+meanent_four_agree
  plt.figure()
  ax = sns.violinplot(data=[np.array(data_meanfa),np.array(data_mean1d),np.array(data_mean2d),np.array(data_mean3d)], cut = 0)   
  ax.set_title("Mean Point Estimate Entropy of Test Points by Different Agreement Levels")   
  ax.set_xlabel("Number of Opposing Pathologist Diagnosis")
  ax.set_ylabel("Entropy") 
  plt.savefig('AgreementMeanPointEntropyFourlevels.png')
  #Median
  plt.figure()
  ax = sns.violinplot(data=[np.array(medianent_zero_agree),np.array(medianent_one_agree),np.array(medianent_two_agree),np.array(medianent_three_agree),np.array(medianent_four_agree),np.array(medianent_five_agree),np.array(medianent_six_agree),np.array(medianent_seven_agree)], cut = 0) 
  ax.set_title("Median Point Estimate Entropy of Test Points by Different Agreement Levels")    
  ax.set_xlabel("Number of Pathologist Diagnosis SSA")
  ax.set_ylabel("Entropy")   
  plt.savefig('AgreementmedianPointEntropyAlllevels.png')
  data_medianfa = medianent_zero_agree+medianent_seven_agree
  data_median1d = medianent_one_agree+medianent_six_agree
  data_median2d = medianent_two_agree+medianent_five_agree
  data_median3d = medianent_three_agree+medianent_four_agree
  plt.figure()
  ax = sns.violinplot(data=[np.array(data_medianfa),np.array(data_median1d),np.array(data_median2d),np.array(data_median3d)], cut = 0)    
  ax.set_title("Median Point Estimate Entropy of Test Points by Different Agreement Levels")   
  ax.set_xlabel("Number of Opposing Pathologist Diagnosis")
  ax.set_ylabel("Entropy")
  plt.savefig('AgreementMedianPointEntropyFourlevels.png')
  #Mode
  plt.figure()
  ax = sns.violinplot(data=[np.array(modeent_zero_agree),np.array(modeent_one_agree),np.array(modeent_two_agree),np.array(modeent_three_agree),np.array(modeent_four_agree),np.array(modeent_five_agree),np.array(modeent_six_agree),np.array(modeent_seven_agree)], cut = 0)    
  ax.set_title("Mode Point Estimate Entropy of Test Points by Different Agreement Levels")    
  ax.set_xlabel("Number of Pathologist Diagnosis SSA")
  ax.set_ylabel("Entropy")
  plt.savefig('AgreementModePointEntropyAlllevels.png')
  data_modefa = modeent_zero_agree+modeent_seven_agree
  data_mode1d = modeent_one_agree+modeent_six_agree
  data_mode2d = modeent_two_agree+modeent_five_agree
  data_mode3d = modeent_three_agree+modeent_four_agree
  plt.figure()
  ax = sns.violinplot(data=[np.array(data_modefa),np.array(data_mode1d),np.array(data_mode2d),np.array(data_mode3d)], cut = 0) 
  ax.set_title("Mode Point Estimate Entropy of Test Points by Different Agreement Levels")   
  ax.set_xlabel("Number of Opposing Pathologist Diagnosis")
  ax.set_ylabel("Entropy")   
  plt.savefig('AgreementModePointEntropyFourlevels.png')
  fa_name = [0] * len(data_fa)
  name1d_ = [1] * len(data_1d)
  name2d_ = [2] * len(data_2d)
  name3d_ = [3] * len(data_3d)
  legend_names = fa_name + name1d_ + name2d_ + name3d_
  dist_entropy = data_fa + data_1d + data_2d + data_3d
  point_est_entropy = data_meanfa + data_mean1d + data_mean2d + data_mean3d
  plt.figure()
  
  plt.scatter(data_fa, data_meanfa)
  plt.scatter(data_1d, data_mean1d)
  plt.scatter(data_2d, data_mean2d)
  plt.scatter(data_3d, data_mean3d)
  plt.legend(["Full Agreement" , "One Opposing", "Two Opposing", "Three Opposing"])
  plt.title("Point Estimate Entropy vs Distribution Entropy")    
  plt.xlabel("Distribution Entropy")
  plt.ylabel("Point Estimate Entropy")
  plt.savefig('EntropyScatterPlot.png')
  
