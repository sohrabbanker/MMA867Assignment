library(readxl)
library(tidyr)
library(tidyverse)
library(mice)
library(ggplot2)
library(sqldf)
library(dplyr)
library(car)
library(estimatr)
library(lubridate)
library(esquisse)
library(caret)
library(stringr)
library(MASS)
library(MLmetrics)
library(Metrics)
library(glmnet)
library(xgboost)
library(gbm)
library(mboost)
library(elasticnet)

bike<-read.csv("C:\\Users\\sohra\\Downloads\\bike-sharing-demand\\train.csv")
str(bike)
summary(bike)
bike_finalanswer<-read.csv("C:\\Users\\sohra\\Downloads\\bike-sharing-demand\\test.csv")

#looking for missing data
md.pattern(bike)
md.pattern(bike_finalanswer)

#check for patterns
pairs(bike)
pairs(bike_finalanswer)

#converting them to factors for regression to recognise easier
bike$season<-as.factor(bike$season)
bike$holiday<-as.factor(bike$holiday)
bike$workingday<-as.factor(bike$workingday)
bike$weather<-as.factor(bike$weather)

#feature engineered hour, day and month in the dataset
bike$datetime<- ymd_hms(bike$datetime)
bike$hour<-hour(bike$datetime)
bike$day<-wday(bike$datetime)
bike$month<-month(bike$datetime, label = T)

bike$hour<-as.factor(bike$hour)
bike$day<-as.factor(bike$day)
bike$month<-as.factor(bike$month)

bike$casual<-NULL #deleted column not required for our test set
bike$registered<-NULL #deleted column not required for our test set

bike$atemp<-ifelse(bike$atemp==0,0.0001,bike$atemp)
#Doing the same for the test database from Kaggle
#converting them to factors for regression to recognise easier
bike_finalanswer$season<-as.factor(bike_finalanswer$season)
bike_finalanswer$holiday<-as.factor(bike_finalanswer$holiday)
bike_finalanswer$workingday<-as.factor(bike_finalanswer$workingday)
bike_finalanswer$weather<-as.factor(bike_finalanswer$weather)

#feature engineered hour, day and month in the dataset
bike_finalanswer$datetime<- ymd_hms(bike_finalanswer$datetime)
bike_finalanswer$hour<-hour(bike_finalanswer$datetime)
bike_finalanswer$day<-wday(bike_finalanswer$datetime)
bike_finalanswer$month<-month(bike_finalanswer$datetime, label = T)

bike_finalanswer$hour<-as.factor(bike_finalanswer$hour)
bike_finalanswer$day<-as.factor(bike_finalanswer$day)
bike_finalanswer$month<-as.factor(bike_finalanswer$month)

bike_finalanswer$casual<-NULL #deleted column not required for our test set
bike_finalanswer$registered<-NULL #deleted column not required for our test set

bike_finalanswer$atemp<-ifelse(bike_finalanswer$atemp==0,0.0001,bike_finalanswer$atemp)
#Split the training dataset 
sample <- sample.int(n=nrow(bike),size=floor(0.8*nrow(bike)),replace = FALSE)

bike_train <- bike[sample,]
bike_test <- bike[-sample,]


#Create a normal multiple regression model using all variables requried for predicition

reg<-lm(count~ ., bike_train)
summary(reg)
test<-predict(reg,bike_test)
rmsle(bike_test$count,test)
rmse(bike_test$count, test)
#Nan
#rmse of 104.6473

par(mfrow=c(1,4))
plot(reg)
plot(density(resid(reg)))

#Create a model using log 
reg_log<-lm(log(count)~datetime+season+holiday+workingday+weather+log(temp)+log(atemp)+humidity+windspeed+hour+day+month, bike_train)
summary(reg_log)
test_log<-exp(predict(reg_log,bike_test))
rmsle(bike_test$count,test_log)
#0.5804
par(mfrow=c(1,4))
plot(reg_log)


#Create interaction variables for both models above
#for log models

reg_log_i<-lm(log(count)~datetime*season+holiday+workingday+weather+log(temp)*log(atemp)+humidity+windspeed+hour*day*month, bike_train)
summary(reg_log_i)
test_reg_log_i<-exp(predict(reg_log_i, bike_test))
rmsle(bike_test$count,test_reg_log_i)
#0.3627

par(mfrow=c(1,4))
plot(reg_log_i)



#Variable Selection
#both
reg_vs1<-step(lm(log(count)~datetime*season+holiday+workingday+weather+log(temp)*log(atemp)+humidity+windspeed+hour*day*month, bike_train), direction = "both")
summary(reg_vs1)
test_reg_vs1<-exp(predict(reg_vs1,bike_test))
rmsle(bike_test$count,test_reg_vs1)
#0.30824

#predicting values for the training set provided by kaggle
test_lasso_kaggle1<-exp(predict(reg_vs1,bike_finalanswer))
write.csv(test_lasso_kaggle1,"C:\\Users\\sohra\\Downloads\\bike-sharing-demand\\New Microsoft Excel Worksheet.csv")

plot(reg_vs1)

#backward
reg_vs2<-step(lm(log(count)~datetime*season+holiday+workingday+weather+log(temp)*log(atemp)+humidity+windspeed+hour*day*month, bike_train), direction = "backward")
summary(reg_vs2)
test_reg_vs2<-exp(predict(reg_vs2,bike_test))
rmsle(bike_test$count,test_reg_vs2)
#0.3304

plot(reg_vs2)

#forward
reg_vs3<-step(lm(log(count)~datetime*season+holiday+workingday+weather+log(temp)*log(atemp)+humidity+windspeed+hour*day*month, bike_train), direction = "forward")
summary(reg_vs3)
test_reg_vs3<-exp(predict(reg_vs3,bike_test))
rmsle(bike_test$count,test_reg_vs3)
#0.3627

##Regularizations

#creating ID column
bike_train$ID <- 1:nrow(bike_train)
bike_test$ID <- 1:nrow(bike_test)+8708
bike_finalanswer$ID <- 1:nrow(bike_finalanswer)

#creating y variable and matrix
#directly creating the training n testing matrices using bike_train, bike_test dataframes
y <- log(bike_train$count)
X_train <- model.matrix(ID ~ datetime*season+hour*day*month+log(temp)*log(atemp)+holiday+
                          workingday+weather+humidity+windspeed,bike_train)[,-1]
X_test <- model.matrix(ID ~ datetime*season+hour*day*month+log(temp)*log(atemp)+holiday+
                         workingday+weather+humidity+windspeed,bike_test)[,-1]
X_validate <- model.matrix(ID ~ datetime*season+hour*day*month+log(temp)*log(atemp)+holiday+
                             workingday+weather+humidity+windspeed,bike_finalanswer)[,-1]

X_train <- cbind(bike_train$ID,X_train)
X_test <- cbind(bike_test$ID,X_test)
X_validate <- cbind(bike_finalanswer$ID,X_validate)

#LASSO model

lasso_reg <- glmnet(x = X_train, y = y, alpha = 1)
plot(lasso_reg, xvar = "lambda")

#selecting best penalty lambda
crossval1 <- cv.glmnet(x = X_train, y = y, alpha = 1)
plot(crossval1)
penalty_lasso_reg <- crossval1$lambda.min #determine optimal penalty parameter, lambda
log(penalty_lasso_reg) #see where it was on the graph
plot(crossval1,xlim=c(-6.3,-5.5),ylim=c(0.1,0.2)) #lets zoom-in

#with optimal penalty
lasso_reg_opt_fit <-glmnet(x = X_train, y = y, alpha = 1, lambda = penalty_lasso_reg)
coef(lasso_reg_opt_fit) #resultant model coefficients

#predicting the performance on the testing set
test_lasso<- exp(predict(lasso_reg_opt_fit, s = penalty_lasso_reg, newx =X_test))
rmsle(bike_test$count,test_lasso)

#predicting values for the training set provided by kaggle
test_lasso_kaggle<-exp(predict(lasso_reg_opt_fit, s = penalty_lasso_reg, newx =X_validate))
write.csv(test_lasso_kaggle,"C:\\Users\\sohra\\Downloads\\bike-sharing-demand\\New Microsoft Excel Worksheet.csv")




#RIDGE model

ridge_reg <- glmnet(x = X_train, y = y, alpha = 0)
plot(ridge_reg, xvar = "lambda")

#selecting best penalty lambda
crossval2 <- cv.glmnet(x = X_train, y = y, alpha =0)
plot(crossval2)
penalty_ridge_reg <- crossval2$lambda.min #determine optimal penalty parameter, lambda
log(penalty_ridge_reg) #see where it was on the graph
plot(crossval2,xlim=c(-6.3,-5.5),ylim=c(0.1,0.2)) #lets zoom-in

#reg 11 with optimal penalty
ridge_reg_opt_fit <-glmnet(x = X_train, y = y, alpha = 0, lambda = penalty_ridge_reg)
coef(ridge_reg_opt_fit) #resultant model coefficients

#predicting the performance on the testing set
test_ridge<- exp(predict(ridge_reg_opt_fit, s = penalty_ridge_reg, newx =X_test))
rmsle(bike_test$count,test_ridge)

