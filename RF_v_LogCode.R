# Emma Collins
# Random Forest vs Logistic Regression
# Portfolio Project, Fall 2020

library(boot)         # for cv.glm()
library(faraway)      # for vif()
library(StepReg)      # for stepwise model selection
library(randomForest) # for random forest
library(ggplot2)      # for ROC plots
library(plotROC)      # fot ROC plots
library(RColorBrewer) # pretty colors for plots

set.seed(22)          # reproducability

### Pretty color prep
display.brewer.all(type = "all", colorblindFriendly = TRUE)
mycol <- brewer.pal(2, "Paired")

### Link to dataset
# https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

sonar <- read.delim("C:\\Users\\Owner\\Desktop\\Portfolio\\RFvLog\\sonar.all-data", header = FALSE, sep = ",")
head(sonar)

summary(sonar)

#rename response column for easier identification
names(sonar)[names(sonar) == "V61"] <- "Y"
colnames(sonar)

# inital look at data for rocks and mines
rocks <- subset(sonar, sonar$Y == "R")
mines <- subset(sonar, sonar$Y == "M")

summary(rocks)
summary(mines)


# univariate plots, for select predictors.  Can we see any patterns?
par(mfrow=c(2,2))
plot(sonar$Y ~ sonar$V10, xlab = "Predictor 10", ylab = "Rock/Mine", col = mycol) #potential relationship
plot(sonar$Y ~ sonar$V25, xlab = "Predictor 25", ylab = "Rock/Mine", col = mycol)
plot(sonar$Y ~ sonar$V39, xlab = "Predictor 39", ylab = "Rock/Mine", col = mycol)
plot(sonar$Y ~ sonar$V58, xlab = "Predictor 58", ylab = "Rock/Mine", col = mycol) #other plots, not so obvious

par(mfrow=c(2,2))
plot(sonar$Y ~ sonar$V7, xlab = "Predictor 7", ylab = "Rock/Mine", col = mycol)   # nothing obvious
plot(sonar$Y ~ sonar$V32, xlab = "Predictor 32", ylab = "Rock/Mine", col = mycol) # low values -> rocks
plot(sonar$Y ~ sonar$V49, xlab = "Predictor 49", ylab = "Rock/Mine", col = mycol) # nothing obvious
plot(sonar$Y ~ sonar$V53, xlab = "Predictor 53", ylab = "Rock/Mine", col = mycol) # nothing obvious 
par(mfrow = c(1,1))

# plot predictors against each other
plot(sonar[,1:10])  # it appears predictors next to each other are pretty similar
plot(sonar[,11:20]) # this trend continues to most of the variables
plot(sonar[,21:30]) 
plot(sonar[,31:40])
plot(sonar[,41:50])
plot(sonar[,51:60]) # less similarity between nearby predictors


############## LOGISTIC REGRESSION ################ 

### Collinearity

cor(sonar[,1:60]) > 0.85
# a little hard to read, try vif

vifmod <- glm(Y~., data = sonar, family = "binomial")
vif(vifmod)
# indicative of lots of collinearity.  
# Since the variables don't have much context, remove largest VIF variable first
# removing variables till all VIF's are under 10

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V17)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V20)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V31)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V33)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V37)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V23)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V28)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V15)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V35)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V23)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V23)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V40)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V11)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V25)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V9)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V18)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V46)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V42)
vif(vifmod)

which.max(vif(vifmod))
vifmod <- update(vifmod, .~.- V48)
vif(vifmod)

# had to remove LOTS of variables.  Yikes.
# there are two variables that still have VIF's over 10 (but less than 11)
#   that are kept in - I'm cautious of removing so many predictors in one go

# create new dataset with non-collinear predictors so it's easier to build the model
newsonar <- sonar[,c(1:8, 10, 12:14, 16, 19, 21, 22, 24, 26, 27, 29, 30, 32, 34, 36, 38, 29, 41, 43, 44, 
                     44, 47, 49:61)]
summary(newsonar)
dim(newsonar)

cor(newsonar[,1:43]) #looks good

### Build Model

# split data

n <- length(sonar$Y) # number of observations

tr <- sample(1:n, size = floor(n*.75))  # note these random selection of observations will also be
t <- (1:n)[-tr]                         # used when building the random forest 

train <- data.frame(newsonar[tr, ])     
test <- data.frame(newsonar[t, ]) 

trainn <- length(tr)
testn <- length(t)

# AIC stepwise selection
fullmod <- glm(Y ~ ., data = train, family = "binomial")

AICmod <- step(fullmod, direction = "both", trace = FALSE)
summary(AICmod)

# Best model has V1, V4, V10, V12, V13, V14, V22, V26, V30, V36, V38, V41, V44, V47, V49
#                V50, V52, V54, V55, V57, V59, V60 as predictors

# BIC stepwise selection
BICmod <- step(fullmod, direction = "both", trace = FALSE, k = log(trainn))
summary(BICmod)

# Best model has V1, V12, V14, V22, V30, V36, V38, V49, V50, V52, V59, V60


# Use cross validation to see which model has lower RMSE 

AICcv <- cv.glm(train, AICmod, K=5)
(AICcv.e <- AICcv$delta[1])

BICcv <- cv.glm(train, BICmod, K=5)
(BICcv.e <- BICcv$delta[1])

# Appears BIC model has lower error, let's try K = 10, and LOO-CV just to be sure

AICcv <- cv.glm(train, AICmod, K=10)
(AICcv.e <- AICcv$delta[1])

BICcv <- cv.glm(train, BICmod, K=10)
(BICcv.e <- BICcv$delta[1])

# BIC still lower, try LOO CV 

AICcv <- cv.glm(train, AICmod, K=trainn)
(AICcv.e <- AICcv$delta[1])

BICcv <- cv.glm(train, BICmod, K=trainn)
(BICcv.e <- BICcv$delta[1])

# BIC better all around

finalmod <- BICmod


############## RANDOM FOREST ################

ftrain <- data.frame(sonar[tr, ])
ftest <- data.frame(sonar[t, ])

# split into test/train sets with same indecies, but with all 60 predictors

rf1 <- randomForest(Y~., data = ftrain) #default 500 trees, not bad, how will increasing trees improve results?
rf1

rf2 <- randomForest(Y~., data = ftrain, ntree = 1000) # still fast calculation, try more trees
rf2

rf3 <- randomForest(Y~., data = ftrain, ntree = 2000) # better than 1000, try more trees
rf3

# random attempt for the ROC graph to work: builds rf on training set then runs it on test set in same function
rf3.5 <- randomForest(x = ftrain[,1:60], y = ftrain[,61], xtest = ftest[,1:60], ytest = ftest[,61], ntree = 2000)
please <- rf3.5$test

rf4 <- randomForest(Y~., data = ftrain, ntree = 3000) # not substantially better prediction for more trees
rf4

# Use rf3 as final model

############### COMPAIRISON ##################

# on test data

logte_pred <- predict.glm(finalmod, newdata = test, type = "response")
log_te <- data.frame(predictor = logte_pred, known.truth = test$Y)
log_te$predicted <- ifelse(log_te$predictor < 0.5,"M","R")
table(log_te$predicted, log_te$known.truth)

rfte_pred <- predict(rf3, newdata = ftest, type = "response", norm.votes = TRUE, predict.all = FALSE, proximity = FALSE, nodes = FALSE)
#rfte_pred <- predict.randomForest(rf3, newdata = ftest, type = "response")
rf_te <- data.frame(predicted = rfte_pred, known.truth = test$Y)
table(rf_te$predicted, rf_te$known.truth)

### True Positive Rate (predicted mine, truly mine / true mines)
log_tpr <- round(sum(log_te$predicted == "M" & log_te$known.truth == "M")/sum(log_te$known.truth == "M"), 3)
rf_tpr <- round(sum(rf_te$predicted == "M" & rf_te$known.truth == "M")/sum(rf_te$known.truth == "M"), 3)

### False Postive Rate (predicted mine, truly rock / true rocks)
log_fpr <- round(sum(log_te$predicted == "R" & log_te$known.truth == "M")/sum(log_te$known.truth == "R"), 3)
rf_fpr <- round(sum(rf_te$predicted == "R" & rf_te$known.truth == "M")/sum(rf_te$known.truth == "R"), 3)

### False Negative Rate  (predicted rock, truly mine / true mines)
log_fnr <- round(sum(log_te$predicted == "M" & log_te$known.truth == "R")/sum(log_te$known.truth == "M"), 3)
rf_fnr <- round(sum(rf_te$predicted == "M" & rf_te$known.truth == "R")/sum(rf_te$known.truth == "M"), 3)
  
  
### Accuracy (correct predictions / all predictions)
log_ac <- round((sum(log_te$predicted == "M" & log_te$known.truth == "M") + 
             sum(log_te$predicted == "R" & log_te$known.truth == "R"))/length(log_te[,2]), 3)
rf_ac <- round((sum(rf_te$predicted == "M" & rf_te$known.truth == "M") + 
             sum(rf_te$predicted == "R" & rf_te$known.truth == "R"))/length(rf_te[,2]), 3)

### Precision (number of mines correctly identified to number of mines identified (correct & incorrect))
log_pr <- round(sum(log_te$predicted == "M" & log_te$known.truth == "M")/sum(log_te$predicted == "M"), 3)
rf_pr <- round(sum(rf_te$predicted == "M" & rf_te$known.truth == "M")/sum(rf_te$predicted == "M"), 3)


### Results in Table
lg <- cbind("logistic regression", log_tpr, log_fpr, log_fnr, log_pr, log_ac)
rff <- cbind("random forest", rf_tpr, rf_fpr, rf_fnr, rf_pr, rf_ac)
results <- data.frame(rbind(lg, rff))
colnames(results) <- c("method", "TPR", "FPR", "FNR", "Precision", "Accuracy")
results


### ROC Curve/AUC


threshold=0.5

df <- data.frame(predictor = logte_pred,known.truth = test$Y)
df$predicted.RM<-ifelse(df$predictor < threshold,0,1)
df$known.truth <- ifelse(df$known.truth=="M", 0, 1)

roc.plot<-ggplot(df, aes(d = known.truth, m = predictor)) + 
  geom_roc(n.cuts = 0) + geom_abline(intercept = 0, slope = 1, linetype="dashed") 

roc.plot + annotate("text", x = .75, y = .25, label = paste("AUC =", round(calc_auc(roc.plot)$AUC, 2)))+
  labs(title = "Logistic Regression ROC Curve", x = "False Postive Fraction", y = "True Positive Fraction")

df2 <- data.frame(predictor = please$votes[,2], known.truth = ftest$Y, predicted.RM <- rfte_pred)
df2$predicted.RM<-ifelse(df2$predictor<threshold,0,1)
df2$known.truth <- ifelse(df2$known.truth=="M", 0, 1)

roc.plot2<-ggplot(df2, aes(d = known.truth, m = predictor)) + 
  geom_roc(n.cuts = 0) + geom_abline(intercept = 0, slope = 1, linetype="dashed") 

roc.plot2 + annotate("text", x = .75, y = .25, label = paste("AUC =", round(calc_auc(roc.plot2)$AUC, 2)))+
  labs(title = "Random Forest ROC Curve", x = "False Postive Fraction", y = "True Positive Fraction")


#### Random stuff #####

#what happens if we just get a log reg with all predictors, no concern for collinearity
wthmod <- glm(Y~., data = ftrain, family = "binomial")
summary(wthmod)
#huge SE, huge p-values. Yikes.

# Training data set comparison values

logtr_pred <- predict.glm(finalmod, newdata = train, type = "response")
log_tr <- data.frame(predictor = logtr_pred, known.truth = train$Y)
log_tr$predicted <- ifelse(log_tr$predictor < 0.5,"M","R")
table(log_tr$predicted, log_tr$known.truth)

rftr_pred <- predict(rf3, newdata = ftrain, type = "response")
rf_tr <- data.frame(predicted = rftr_pred, known.truth = train$Y)
table(rf_tr$predicted, rf_tr$known.truth)

### True Positive Rate (predicted mine, truly mine / true mines)
tlog_tpr <- round(sum(log_tr$predicted == "M" & log_tr$known.truth == "M")/sum(log_tr$known.truth == "M"), 3)
trf_tpr <- round(sum(rf_tr$predicted == "M" & rf_tr$known.truth == "M")/sum(rf_tr$known.truth == "M"), 3)

### False Postive Rate (predicted mine, truly rock / true rocks)
tlog_fpr <- round(sum(log_tr$predicted == "R" & log_tr$known.truth == "M")/sum(log_tr$known.truth == "R"), 3)
trf_fpr <- round(sum(rf_tr$predicted == "R" & rf_tr$known.truth == "M")/sum(rf_tr$known.truth == "R"), 3)

### False Negative Rate  (predicted rock, truly mine / true mines)
tlog_fnr <- round(sum(log_tr$predicted == "M" & log_tr$known.truth == "R")/sum(log_tr$known.truth == "M"), 3)
trf_fnr <- round(sum(rf_tr$predicted == "M" & rf_tr$known.truth == "R")/sum(rf_tr$known.truth == "M"), 3)


### Accuracy (correct predictions / all predictions)
tlog_ac <- round((sum(log_tr$predicted == "M" & log_tr$known.truth == "M") + 
                   sum(log_tr$predicted == "R" & log_tr$known.truth == "R"))/length(log_tr[,2]), 3)
trf_ac <- round((sum(rf_tr$predicted == "M" & rf_tr$known.truth == "M") + 
                  sum(rf_tr$predicted == "R" & rf_tr$known.truth == "R"))/length(rf_tr[,2]), 3)

### Precision (number of mines correctly identified to number of mines identified (correct & incorrect))
tlog_pr <- round(sum(log_tr$predicted == "M" & log_tr$known.truth == "M")/sum(log_tr$predicted == "M"), 3)
trf_pr <- round(sum(rf_tr$predicted == "M" & rf_tr$known.truth == "M")/sum(rf_tr$predicted == "M"), 3)

#table of results
tlg <- cbind("logistic regression", tlog_tpr, tlog_fpr, tlog_fnr, tlog_pr, tlog_ac)
trff <- cbind("random forest", trf_tpr, trf_fpr, trf_fnr, trf_pr, trf_ac)
tresults <- data.frame(rbind(tlg, trff))
colnames(tresults) <- c("method", "TPR", "FPR", "FNR", "Precision", "Accuracy")
tresults

