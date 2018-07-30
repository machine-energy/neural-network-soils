## Predicting sine function using an MLP Neural Network ----
# Overview: Neural network model in R to predict soil conductivity as a function of pH and elemental composition (with reference to https://datascienceplus.com/fitting-neural-network-in-r/)
# Dataset: Soils (https://vincentarelbundock.github.io/Rdatasets/datasets.html)
# Author: Dr Gobinath Pillai Rajarathnam
# Google Scholar: https://scholar.google.com.au/citations?user=7mbZHrcAAAAJ&hl=en
# GitHub: https://github.com/machine-energy
# LinkedIn: https://au.linkedin.com/in/gobinath-rajarathnam-0364a910b

## Initialisation and starting timer ----
library(tictoc) # load timer package
tic("Time for code to run") # start timer for code runtime
setwd("~/Downloads") # Set working directory
library(boot) # load the "boot" package 
library(dplyr) # load the "dplyr" package 
library(neuralnet) # load the "neuralnet" package 
library(plyr) # load the "plyr" package 
library(readr) # to import csv files
library(data.table)

# load the dataset
data <- read_csv("Soils.csv") # import dataset
data <- subset(data[7:ncol(data)]) # subset data for parameters of interest

# setting global variables
nodes1stlayer <- 5 # number of nodes in 1st layer (closer to input)
nodes2ndlayer <- 3 # number of nodes in 2nd layer (closer to output)
kfold <- 5 # many-fold cross-validation to increase robustness of model (in this case 5 times)
holdback <- 0.7 # how much of the data to use in training set (in this case, 70%)
set.seed(10) # set the seed (in this case 10) for reproducibility of results
cv.error <- NULL # initialise the cross-validation error

## Preparing and fitting the neural network ----
# scaling the data for use in the neural network
max.values <- apply(data, 2, max) # setting the scale
min.values <- apply(data, 2, min) # setting the scale
scaled <- as.data.frame(scale(data, center = min.values, scale = max.values - min.values)) # scaling the data

# setup of neural network formula
f <- as.formula(paste("Conduc ~ pH + N + Dens + P + Ca + Mg + K + Na"))
# create and start a progress bar to monitor model development
pbar <- create_progress_bar('text')
pbar$init(kfold)

# cross-validation loop
for(i in 1:kfold){
  index <- sample(1:nrow(data),round(holdback*nrow(data))) # indexing the data
  train.cv <- scaled[index,] # create a training set using the indexed sample
  test.cv <- scaled[-index,] # create a testing set using complement of the indexed sample
  nn <- neuralnet(f, data = train.cv, hidden = c(nodes1stlayer,nodes2ndlayer), linear.output = TRUE) # training the neural network model
  pr.nn <- compute(nn,test.cv[,1:8]) # doing new predictions using the now-trained neural network
  pr.nn <- pr.nn$net.result*(max(data$Conduc)-min(data$Conduc))+min(data$Conduc) # re-scaling calculated results back up
  test.cv.r <- (test.cv$Conduc)*(max(data$Conduc)-min(data$Conduc))+min(data$Conduc) # re-scaling test data back up
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv) # keep running log of cross-validation errors
  pbar$step() # updates the progress bar
}

## model results extraction and post-processing ----
mean(cv.error) # returns the mean error from cross-validation runs
pr.nn <- compute(nn,scaled[,1:8]) # doing predictions on the full dataset using the now-trained neural network
pr.nn <- pr.nn$net.result*(max(data$Conduc)-min(data$Conduc))+min(data$Conduc) # re-scaling calculated results back up
collated.results <- merge(data, pr.nn, by=0) # merge dataframes while leaving out the "Row.names" column from the merge
setnames(collated.results, "V1", "Predicted Conduc") # rename predicted value column
write.csv(collated.results, "Soil NN Results.csv") # write results to a csv file for collaboration/sharing/records

# plot neural network
plot(nn)

toc() # finish the timer for this code
