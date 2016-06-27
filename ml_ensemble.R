###Libraries
#load these packages
pacman::p_load(ggplot2, RColorBrewer, dplyr, scales, Hmisc, GGally,
               caret, caretEnsemble)

#check that all packages loaded sucessfully
pacman::p_loaded(ggplot2, RColorBrewer, scales, dplyr, Hmisc, GGally,
                 caret, caretEnsemble)

#Read the data
lc_loans<-read.csv("lc_loans.csv")


#create outcome variables; 
# very_good (A): duration 12 or above months and positive cummulative return
# bad (B): negative cummulative return
lc_loans <- lc_loans %>%
    mutate(outcome=as.factor(ifelse(duration>=12 & cum_r>.08, 
                                    "very_good", "bad"))) %>%
    filter(!is.na(outcome)) %>%
    filter(home_ownership!="ANY")

describe(lc_loans$outcome)


lc_loans$grade<-as.numeric(lc_loans$grade)
lc_loans$sub_grade<-as.numeric(lc_loans$sub_grade)
lc_loans$home_ownership<-as.numeric(lc_loans$home_ownership)
lc_loans$verification_status<-as.numeric(lc_loans$verification_status)
lc_loans$purpose<-as.numeric(lc_loans$purpose)
lc_loans$addr_state<-as.numeric(lc_loans$addr_state)

#use the following features. Remove features with leakage
features<-c("loan_amnt", "term2","int_rate","installment",
            "emp_length","annual_inc","grade",
            "dti","delinq_2yrs","inq_last_6mths",
            "open_acc","pub_rec","revol_bal","revol_util","total_acc",
            "tot_cur_bal","open_acc_6m","total_bal_il","all_util",
            "total_rev_hi_lim","inq_last_12m","avg_cur_bal",
            "pub_rec_bankruptcies",
            "tax_liens","tot_hi_cred_lim","total_bal_ex_mort","issue_year",
            "credit_years","mean_fico","loan_to_inc","comment_n",
            "comment_exists","dollar_signs","words_n",
            "sentiment_syuzhet","sub_grade","home_ownership",
            "verification_status","purpose","addr_state")

#find the index for the features
index<- which(names(lc_loans) %in% features)


#deal with missing data
na_count <-sapply(lc_loans[,index], function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count$name<-rownames(na_count)
lc_loans[,index][is.na(lc_loans[,index])] <- 0

#create test and training sets (60:40)
set.seed(123)
inTrain <- createDataPartition(lc_loans$outcome, p = .1, list = FALSE)
training <- lc_loans[ inTrain,]
testing <- lc_loans[-inTrain,]


#check the proportion for each class
table(training$outcome)/nrow(training)

#check the proportion for each class
table(testing$outcome)/nrow(testing)
#looks good





#cross-validation procedure
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 5, 
                savePredictions="final",
                allowParallel = T,
                summaryFunction=twoClassSummary,
                classProbs=TRUE) 


#run several models and see which ones show high correlation
library(doParallel)
cl <- makeCluster(3)
registerDoParallel(cl)

model_list <- caretList(
    y=training$outcome,
    x=training[,index],
    trControl=cv.ctrl,
    methodList=c("rf","xgbTree","rpart","gbm","C5.0"),
    continue_on_fail=TRUE,
    metric="Kappa")


#greedy_enemble models
greedy_ensemble <- caretEnsemble(
    model_list, 
    metric="ROC",
    trControl=trainControl(
        number=5,
        summaryFunction=twoClassSummary,
        classProbs=TRUE
    ))
summary(greedy_ensemble)
beepr::beep(8)

stopCluster(cl)
#it doesn't look like ensembling will do much of anything, since all models seem to be correlated with each other
modelCor(resamples(model_list))


#make predictions
testing$ensample_pred <- predict(greedy_ensemble, 
                                 newdata=testing[,index])
testing$rpart_pred <- predict(model_list$rpart, 
                                  newdata=testing[,index])
testing$gbm <- predict(model_list$gbm, 
                               newdata=testing[,index])
testing$rf <- predict(model_list$rf, 
                       newdata=testing[,index])
testing$C5.0 <- predict(model_list$C5.0, 
                       newdata=testing[,index])
testing$xgbTree <- predict(model_list$xgbTree, 
                      newdata=testing[,index])



confusionMatrix(testing$ensample_pred,testing$outcome)
confusionMatrix(testing$rpart_pred,testing$outcome)
confusionMatrix(testing$gbm,testing$outcome)
confusionMatrix(testing$rf ,testing$outcome)
confusionMatrix(testing$C5.0,testing$outcome)
confusionMatrix(testing$xgbTree,testing$outcome)

 

# ####return measures predicted for ensamble
# library(dplyr)
# 
# testing %>% filter(rf>.10) %>% 
#     summarise_each(funs(mean,median,n()), cum_r, ann_r)
# 
# 
# testing %>% filter(ensample_pred>.10) %>%
#     summarise_each(funs(mean,median,n()), cum_r, ann_r)
# 
# 
# testing %>% filter(xgbTree>.10,n()) %>%
#     summarise_each(funs(mean,median,n()), cum_r, ann_r)
# 



####return measures predicted for each model
detach("package:plyr", unload=TRUE)
library(dplyr)
results<-select(testing, cum_r, ensample_pred, rpart_pred,
                gbm, rf, C5.0, xgbTree)

library(tidyr)
results2<-results %>% gather(model,outcome,-cum_r)


results3<-results2 %>%  group_by(model, outcome) %>%
    summarise(mean_r=mean(cum_r),
              median_r=median(cum_r))

# model   outcome        mean     median
# (chr)     (chr)       (dbl)      (dbl)
# 1           C5.0       bad -0.06152702 0.05792093
# 2           C5.0 very_good  0.06625312 0.13387030
# 3  ensample_pred       bad -0.05863269 0.06092032
# 4  ensample_pred very_good  0.07571062 0.13769224
# 5            gbm       bad -0.05578675 0.06291469
# 6            gbm very_good  0.07670385 0.14098069
# 7             rf       bad -0.05856887 0.06043307
# 8             rf very_good  0.07255531 0.13640748
# 9     rpart_pred       bad -0.04821497 0.06652786
# 10    rpart_pred very_good  0.08004657 0.14311501
# 11       xgbTree       bad -0.05829626 0.06143000
# 12       xgbTree very_good  0.07588089 0.13811625


#filter only the very_good results
results3<-results3 %>% filter(outcome=="very_good") %>% select(-c(outcome))

#add the baseline mean and median cum_r and ann_r for all loans
model<-c("baseline")
mean_r<-c(mean(lc_loans$cum_r))
median_r<-c(median(lc_loans$cum_r))


baseline.data <- data.frame(model,mean_r, median_r)

results3<-rbind(results3, baseline.data)

results4<-results3 %>% 
    gather(return_type,return, mean_r:median_r)

ggplot(results4, aes(return_type, return)) +   
    geom_bar(aes(fill = model), position = "dodge", stat="identity")

#save the current session; big file
session::save.session("lending_ml.Rda")
write.csv(results4,"results4.csv")

