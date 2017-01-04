library(data.table)
library(xgboost)
library(Metrics)
library(Matrix)
library(mice)
library(dplyr)

TRAIN='train.csv'
TEST='test.csv'
SUBMISSION= "sample_submission.csv"

#load data
train=fread(TRAIN,showProgress = T)
test=fread(TEST,showProgress = T)
y_train=train$SalePrice

#Remove Id since of no use
train$Id=NULL
train$SalePrice=NULL
test$Id=NULL

#Row binding train & test set for feature engineering
train_test = rbind(train, test)
ntrain=nrow(train)

features=names(train)

#convert character into integer
for(f in features){
  if(class(train_test[[f]])=="character"){
    levels=sort(unique(train_test[[f]]))
    train_test[[f]]=as.integer(factor(train_test[[f]],levels = levels))
  }
}

#feature to exclude
features_to_drop=  c("")
  
  
  #c("Utilities","LotFrontage","Alley","MasVnrType","MasVnrArea","BsmtQual",
  #                  "BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
  #                  "Electrical","FireplaceQu","GarageType","GarageYrBlt",
  #                  "GarageFinish","GarageQual","GarageCond","PoolQC",
  #                  "Fence","MiscFeature")


#splitting whole data back again
train_x=train_test[1:ntrain,-features_to_drop,with= FALSE]
test_x=train_test[(ntrain+1):nrow(train_test),-features_to_drop,with= FALSE]


#missing values imputation with mice
set.seed(144)


train_x[] = lapply(train_x, as.numeric)
test_x[]=lapply(test_x, as.numeric)

dtrain=xgb.DMatrix(as.matrix(train_x),label= y_train)
dtest=xgb.DMatrix(as.matrix(test_x))

#xgboost parameters
xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.5,
  eta = 0.015, 
  objective = 'reg:linear',
  max_depth = 6,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)

xg_eval_mae = function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

best_n_rounds=1000 # try more rounds

#train data
gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))
submission=fread(SUBMISSION,colClasses = c("integer","numeric"))
submission$SalePrice=predict(gb_dt,dtest)
write.csv(submission,"test_xgb.csv",row.names = FALSE)
