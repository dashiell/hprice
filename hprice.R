library(data.table)
library(caret)
library(plyr)



setwd("~/GitHub/hprice")

training = fread('train.csv', stringsAsFactors = TRUE)
trows = nrow(training)

test = fread('test.csv', stringsAsFactors = TRUE)

data = rbind(training, test, fill=TRUE)
rm(training, test)

data[, Id := NULL]

#get indexes for all the columns that are of type factor
charIdx = which(sapply(data, is.factor))
for(i in charIdx) {
  set(data, i=which(is.na(data[[i]])), j=i, value="MISSING")
}
#get indexes for all the columns that are of type factor
charIdx = which(sapply(data, is.integer))
for(i in charIdx) {
  set(data, i=which(is.na(data[[i]])), j=i, value=-1)
}

#Make sure there are no NA vals in the data set
sapply(dt.train,function(x) sum(is.na(x)))

training = data[1:trows, ]
dt.test = data[-seq_len(trows), ]

inTraining = createDataPartition(training$SalePrice, p=3/4, list=FALSE)
dt.train = training[inTraining,]
dt.valid = training[-inTraining,]

rm(training, inTraining)

#nzv = nearZeroVar(dt.train, saveMetrics = TRUE)
#nzv[nzv$nzv,]
#dropCols = names(dt.train)[nzv$nzv]
## from dropCols, keep these features..
#removeFromDrop = c("KitchenAbvGr", "PoolArea", "PoolQC", "LowQualFinSF", "Functional")
#dropCols[!dropCols %in% removeFromDrop]
#dt.train[, (dropCols) := NULL]

fitControl = trainControl(method="cv", repeats=5)
#fitControl = trainControl(method="none", classProbs=FALSE, verboseIter = TRUE)
tuneGrid = expand.grid(n.trees=100, interaction.depth=3, shrinkage=0.1, n.minobsinnode=10)

gbm_fitted = train(SalePrice ~., data=dt.train, method="gbm", trControl=fitControl, tuneGrid=tuneGrid, metric="RMSE")
gbm_predict = predict(gbm_fitted, dt.valid) 
( rmse = sqrt(mean( (dt.valid$SalePrice - gbm_predict)^2)) )

#glm_fitted = train(SalePrice ~., data=dt.train, method="glm", trControl=fitControl)
#glm_predict = predict(glm_fitted, dt.valid)
#( rmse = sqrt(mean( (dt.valid$SalePrice - glm_predict)^2)) )

# Impute TotalBsmtSF BsmtExposure

# lot frontage is probably important to price; impute it with rpart
#col.pred = c("MSZoning", "MSSubClass", "LotFrontage", "LotArea", "Alley", "LotShape", "BldgType", "HouseStyle", "YrSold")
#dt.frontage_subset = df.train %>% select(one_of(col.pred)) # use one_of to get columns by name
#dt.known_frontage = df.frontage_subset %>% filter(!is.na(LotFrontage)) 
#dt.unknown_frontage = df.frontage_subset %>% filter(is.na(LotFrontage)) 
#fitted_frontage = rpart(LotFrontage ~., data=df.known_frontage)
#predict_frontage = as.integer(melt(predict(fitted_frontage, df.unknown_frontage))$value)

#dt.train[is.na(LotFrontage), LotFrontage := predict_frontage] 




