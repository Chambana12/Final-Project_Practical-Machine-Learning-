# Final-Project_Practical-Machine-Learning-
Final Project - Practical Machine Learning

library(ggplot2)
library(caret)
library(rpart)
library(rattle)
library(randomForest)
library(dplyr)
library(corrplot)
library(gbm)

## CLEANING DATA
# split training data into 2 separate sets: myTraining and myTesting 
Training <- read.csv("/Users/markomadunic/Desktop/Data/PRACTICAL MACHINE LEARNING /WEEK 4/data/pml-training.csv")
Testing <- read.csv("/Users/markomadunic/Desktop/Data/PRACTICAL MACHINE LEARNING /WEEK 4/data/pml-testing.csv")

inTrain <- createDataPartition(y = Training$classe, p = 0.7, list =F)
myTraining <- Training[inTrain, ]
myTesting <- Training[-inTrain, ]

# create local data sets of myTraining and myTesting
myTraining <- tbl_df(myTraining)
myTesting <- tbl_df(myTesting)
glimpse(myTraining)
glimpse(myTesting)

# only 269 complete cases - many NAs
sum(complete.cases(myTraining))
# [1] 269

# breakdown of the classifying variable
table(myTraining$classe)
A    B    C    D    E 
3906 2658 2396 2252 2525 

# REMOVE zero covariates 
# zeroVar - a vector of logicals for whether the predictor has only one distinct value
# nzv	- a vector of logicals for whether the predictor is a near zero variance predictor
nsv <- nearZeroVar(x = myTraining, saveMetrics = T)

# turn row names into an explicit chr variable
nsv <- nsv %>% add_rownames("variables")

nsv_false <- nsv %>% filter(nzv==FALSE)
nzv_vars <- as.character(nsv_false$variables)

# select data frames with only columns where nzv==FALSE
myTraining <- myTraining %>% 
    select(one_of(nzv_vars))

myTesting <- myTesting %>%
    select(one_of(nzv_vars))

dim(myTraining)
[1] 13737    53
dim(myTesting)
[1] 5885   53

# remove first variable X - row index
myTraining <- myTraining[,-1]
myTesting <- myTesting[,-1]

## sapply on myTraining and calculate if % of NAs in each feature is greater than 0.95 respectively. 
# rationale: variables that contain in excess of 95% NA elements do not contribute any predictive power
NA.95 <- sapply(myTraining, function(x) {mean(is.na(x))}) > 0.95

# results show that 
myTraining <- myTraining[, NA.95==FALSE]
myTesting <- myTesting[, NA.95==FALSE]

dim(myTraining)
[1] 13737    57
dim(myTesting)
[1] 5885   57

# remove first 5 variables that serve as identifiers, not predictors
myTraining <- myTraining[,-(1:5)]
myTesting <- myTesting[,-(1:5)]

## CREATE CORRELATION MATRIX
cor_table <- cor(myTraining[,-53])

# create correlation plot by using method of "first principal component order"
corrplot(cor_table, type="lower", method ="color", order="FPC", tl.cex=0.65, tl.col = "black")

## USING DECISION TREES TO PREDICT
# use "rpart" analysis - recursive partitioning and regression trees
rm(modRPART1)
set.seed(1234)
modRPART <- rpart(classe ~., data=myTraining, method="class")
print(modRPART$frame)
fancyRpartPlot(modRPART)

# E.g., 2 results stand out: If roll_belt > 130 we predict with 99% certainty E class 
#  If roll_belt < 130 and pitch_forearm < -34, we predict A class with 99% certaainty
# predicting new values with rPART 

predRPART <- predict(modRPART, newdata = myTesting, type = "class")
CM_RPART <- confusionMatrix(predRPART, myTesting$classe)
CM_RPART

# results mapped on a plot matrix
plot(CM_RPART$table, CM_RPART$byClass, main="Overall Accuracy = 0.7499", color="light blue")

## USING RANDOM FOREST TO PREDICT
set.seed(1234)
modRF <- randomForest(classe ~., data = myTraining)
predRF <- predict(modRF, newdata = myTesting, type = "class")
CM_RF <- confusionMatrix(predRF, myTesting$classe)
CM_RF

set.seed(1234)
modRF1 <- randomForest(classe ~., data = myTraining)
predRF <- predict(modRF, newdata = myTesting, type = "class")
CM_RF <- confusionMatrix(predRF, myTesting$classe)
CM_RF

# results mapped on a plot matrix
plot(CM_RF$table, CM_RF$byClass, main="Overall Accuracy RF = 0.9969", color="light green")

## RF 
ctrRF <- trainControl(method="cv", number=3, verboseIter = F)
modRF1 <- train(classe~., data=myTraining, method="rf", trControl= ctrRF)
modRF1$finalModel
plot(modRF)

# GENERALIZED BOOSTSED REGRESSION - 
# n.trees = 150 (iterations)
# accuracy of the final model = 95.96%
# 52 predictors of which 41 had non-zero influence
set.seed(1234)
ctrGBM <- trainControl(method="repeatedcv", number=5, repeats=1)
modGBM <- train(classe~., method="gbm", data=myTraining, verbose=F, trControl=ctrGBM)

fin_mod_gbm <- modGBM$finalModel

# predict on the out-of-sample values 
predict_gbm <- predict(modGBM, newdata = myTesting)
CM_GBM <- confusionMatrix(predict_gbm, myTesting$classe)
CM_GBM

plot(modGBM)
fin_mod_gbm

# accuracy results mapped on a plot matrix
plot(CM_GBM$table, col=CM_GBM$byClass, main="Overall Accuracy GBM = 0.9596", color="pink")


### FINAL PREDICTION TEST on testing data set
# based on prediction accuracy of 3 models, I select Random Forrest model to use on the validation set
validat_test <- predict(modRF, newdata = Testing, type="class")
validat_test

validat_pred <- data.frame(
    caseID=Testing$problem_id,
    prediction=validat_test)

validat_pred
