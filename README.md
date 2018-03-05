# Final Project: Practical Machine Learning

library(ggplot2)
library(caret)
library(rpart)
library(rattle)
library(randomForest)
library(dplyr)
library(corrplot)
library(gbm)

## CLEANING DATA
### Split training data into 2 separate sets: myTraining and myTesting 
Training <- read.csv("/Users/markomadunic/Desktop/Data/PRACTICAL MACHINE LEARNING /WEEK 4/data/pml-training.csv")
Testing <- read.csv("/Users/markomadunic/Desktop/Data/PRACTICAL MACHINE LEARNING /WEEK 4/data/pml-testing.csv")

inTrain <- createDataPartition(y = Training$classe, p = 0.7, list =F)
myTraining <- Training[inTrain, ]
myTesting <- Training[-inTrain, ]

### Create local data sets of myTraining and myTesting
myTraining <- tbl_df(myTraining)
myTesting <- tbl_df(myTesting)
glimpse(myTraining)
glimpse(myTesting)

### Only 269 complete cases - many NAs
sum(complete.cases(myTraining))
[1] 269

### Breakdown of the classifying variable
table(myTraining$classe)
A    B    C    D    E 
3906 2658 2396 2252 2525 

### REMOVE zero covariates 
### zeroVar - a vector of logicals for whether the predictor has only one distinct value
### nzv	- a vector of logicals for whether the predictor is a near zero variance predictor
nsv <- nearZeroVar(x = myTraining, saveMetrics = T)

### Turn row names into an explicit chr variable
nsv <- nsv %>% add_rownames("variables")

nsv_false <- nsv %>% filter(nzv==FALSE)
nzv_vars <- as.character(nsv_false$variables)

### Select data frames with only columns where nzv==FALSE
myTraining <- myTraining %>% 
    select(one_of(nzv_vars))

myTesting <- myTesting %>%
    select(one_of(nzv_vars))

dim(myTraining)
[1] 13737    53
dim(myTesting)
[1] 5885   53

### Remove first variable X - row index
myTraining <- myTraining[,-1]
myTesting <- myTesting[,-1]

### Sapply on myTraining and calculate if % of NAs in each feature is greater than 0.95 respectively. 
### Rationale: variables that contain in excess of 95% NA elements do not contribute any predictive power
NA.95 <- sapply(myTraining, function(x) {mean(is.na(x))}) > 0.95

### Results show that 
myTraining <- myTraining[, NA.95==FALSE]
myTesting <- myTesting[, NA.95==FALSE]

dim(myTraining)
[1] 13737    57
dim(myTesting)
[1] 5885   57

### Remove first 5 variables that serve as identifiers, not predictors
myTraining <- myTraining[,-(1:5)]
myTesting <- myTesting[,-(1:5)]

## Create CORRELATION MATRIX
cor_table <- cor(myTraining[,-53])

### Create correlation plot by using method of "first principal component order"
corrplot(cor_table, type="lower", method ="color", order="FPC", tl.cex=0.65, tl.col = "black")

![image](https://user-images.githubusercontent.com/34659183/36941875-30d6d172-1f1a-11e8-9c24-781eda03ad51.png)

## To predict "classe" variable, I will use 3 classification methods. 
## 1. FOR THE FIRST METHOD I USE DECISION TREES
### Use "rpart" analysis - recursive partitioning and regression trees
rm(modRPART1)
set.seed(1234)
modRPART <- rpart(classe ~., data=myTraining, method="class")
print(modRPART$frame)
fancyRpartPlot(modRPART)

![image](https://user-images.githubusercontent.com/34659183/36958884-bfcb38b2-1ff3-11e8-908e-1b0fe85793e8.png)

Overall Statistics
                                          
               Accuracy : 0.7499          
                 95% CI : (0.7386, 0.7609)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6836          
 Mcnemar's Test P-Value : < 2.2e-16       



# E.g., 2 results stand out: If roll_belt > 130 we predict with 99% certainty E class 
#  If roll_belt < 130 and pitch_forearm < -34, we predict A class with 99% certaainty
# predicting new values with rPART 

predRPART <- predict(modRPART, newdata = myTesting, type = "class")
CM_RPART <- confusionMatrix(predRPART, myTesting$classe)
CM_RPART

Overall Statistics
                                         
               Accuracy : 0.9968         
                 95% CI : (0.995, 0.9981)
    No Information Rate : 0.2845         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9959         
 Mcnemar's Test P-Value : NA             

# results mapped on a plot matrix
plot(CM_RPART$table, CM_RPART$byClass, main="Overall Accuracy = 0.7499", color="light blue")

![image](https://user-images.githubusercontent.com/34659183/36959325-50fab4fa-1ff6-11e8-9a4d-3d88852bcca3.png)


# 2. USING RANDOM FOREST TO PREDICT
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

![image](https://user-images.githubusercontent.com/34659183/36959279-17f4e1bc-1ff6-11e8-8457-ebe7b30ae4c2.png)


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

Overall Statistics
                                          
               Accuracy : 0.9648          
                 95% CI : (0.9598, 0.9694)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9555          
 Mcnemar's Test P-Value : 0.0005234       

# predict on the out-of-sample values 
predict_gbm <- predict(modGBM, newdata = myTesting)
CM_GBM <- confusionMatrix(predict_gbm, myTesting$classe)
CM_GBM

plot(modGBM)
![image](https://user-images.githubusercontent.com/34659183/36959527-6b1d67f0-1ff7-11e8-892b-fd501e4b04a5.png)

fin_mod_gbm

# accuracy results mapped on a plot matrix
plot(CM_GBM$table, col=CM_GBM$byClass, main="Overall Accuracy GBM = 0.9596", color="pink")

![image](https://user-images.githubusercontent.com/34659183/36959645-f4184296-1ff7-11e8-9691-852df3c5c248.png)

### FINAL PREDICTION TEST on testing data set
# based on prediction accuracy of 3 models, I select Random Forrest model to use on the validation set
validat_test <- predict(modRF, newdata = Testing, type="class")
validat_test

validat_pred <- data.frame(
    caseID=Testing$problem_id,
    prediction=validat_test)

validat_pred
   caseID prediction
1       1          B
2       2          A
3       3          B
4       4          A
5       5          A
6       6          E
7       7          D
8       8          B
9       9          A
10     10          A
11     11          B
12     12          C
13     13          B
14     14          A
15     15          E
16     16          E
17     17          A
18     18          B
19     19          B
20     20          B
