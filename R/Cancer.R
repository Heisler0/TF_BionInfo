# Wisconsin Data Set, Cancer
library(rpart)
library(rpart.plot)
library(randomForest)
library(plyr)
library(e1071)
setwd("C:\\Users\\Ryan\\TF_BionInfo\\R")

csv <- read.csv("data.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
csv[,33] = NULL # remove the last column, there is no data in there
csv[,1] = NULL # remove the id, since it is obviously not going to power our model

count(csv, c('diagnosis'))  # good size, 357 benign, 212 maligant


csv$diagnosis = sapply(csv$diagnosis, function(s)
{
  if (s=="M") # maligant
  {
    return(0)
  }
  else
  {
    return(1) # benign
  }
    
})

csv$diagnosis = as.factor(csv$diagnosis) # diagnosis is not continous

# all our other values look okay...we might have to normalize later
# check correlation to diagnosis
# looks like radius_mean is close
#c = cor(csv, csv$diagnosis)


# predict the diagnosis based on these attributes

# divide our data into training and testing (70%, 30%)
set.seed(12) # set seed to allow for reproducability
train=sample(1:nrow(csv), nrow(csv) / 2)
test=-train
trainingData=csv[train,]
testingData=csv[test,]
formula = trainingData$diagnosis~fractal_dimension_mean+texture_mean+perimeter_mean+area_mean+smoothness_mean+compactness_mean+concavity_mean+concave.points_mean+symmetry_mean+fractal_dimension_mean+radius_se+texture_se+perimeter_se+area_se+smoothness_se+compactness_se+concavity_se+concave.points_se+symmetry_se+fractal_dimension_se+radius_worst+texture_worst+perimeter_worst+area_worst+smoothness_worst+compactness_worst+concavity_worst+concave.points_worst+symmetry_worst+fractal_dimension_worst
 


#1: Random Forest
rf_model = randomForest(trainingData$diagnosis ~ ., data=trainingData)
summary(rf_model)
rf_pred = predict(rf_model,testingData)
rf_ans = csv$diagnosis[test]
mean(rf_pred == rf_ans)
# 96.84% accuract with random forest

# 2. Logistic Regression
logit = glm(diagnosis ~ ., data = trainingData, family = "binomial", control=list(maxit=200))
testingData$guessDiagnosis = predict(logit, newdata = testingData[,1:31], type="response")
testingData$guessDiagnosis = sapply(testingData$guessDiagnosis, function(n)
{
  if (n <= 0.5)
  {
    return(0)
  }
  else
  {
    return(1)
  }
})
count(testingData, c('guessDiagnosis', 'diagnosis'))
# accuracy is
(95 + 169) / (95 + 10 + 11 + 169)
# 92% accuracy with logistic regression

#3. Naive Bayes
testingData$guessDiagnosis = NULL # remove from logistic regression
nb_model = naiveBayes(diagnosis ~ ., data = trainingData)
nb_pred = predict(nb_model, newdata = testingData[,1:31])
nb_ans = csv$diagnosis[test]
mean(nb_pred == nb_ans)
# 95.08% correct with naive bayes
