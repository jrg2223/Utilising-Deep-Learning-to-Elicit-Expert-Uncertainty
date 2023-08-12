# --- Load functions --- #
from TrainingEvaluatingandModelDiagnosticFunctions import train_pytorch_cnn_dropout_test,agreementreporting,elicit_distribution
from LoadDataScript import dataset, dataset_testing

# -- Select key parameter values -- #
epochs = 100
num_samples = 10
batch_size = 32
img_size = 224

print("Start Training")
net,testing_set = train_pytorch_cnn_dropout_test(1, dataset, batch_size, epochs, num_samples,dataset_testing)
print("Training Complete")
print("Start Agreement Stats")
agreementreporting(testing_set,net)
print("Agreement Stats Finished")
print("Start Elicitating new Observation Plots")
elicit_distribution(net,dataset_testing)
print("Finish Elicitating new Observation Plots")
print("Finished ALL")