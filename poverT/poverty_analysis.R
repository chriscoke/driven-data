
########################################################

library(data.table)
library(dplyr)
library(xgboost)


########################################################
# Import Data

ai = fread("~/../Downloads/A_indiv_train.csv", data.table=F) # training set at the individial level, country A
ah = fread("~/../Downloads/A_hhold_train.csv", data.table=F) # training set at the household level, country A
bi = fread("~/../Downloads/B_indiv_train.csv", data.table=F)
bh = fread("~/../Downloads/B_hhold_train.csv", data.table=F)
ci = fread("~/../Downloads/C_indiv_train.csv", data.table=F)
ch = fread("~/../Downloads/C_hhold_train.csv", data.table=F) # training set at the household level, country C


########################################################
# Explore the data a little

# class splits

table(ai$poor)
table(ah$poor)
table(bi$poor)
table(bh$poor)
table(ci$poor)
table(ch$poor)

# shared variables across sets

summary(names(ah) %in% names(bh))
summary(names(ah) %in% names(ch))
summary(names(bh) %in% names(ch))
summary(names(ai) %in% names(bi))
summary(names(ai) %in% names(ci))
summary(names(bi) %in% names(ci))


########################################################
# Find numerical variables

# find worthy numeric columns (all others will be categorical)

numeric.candidates.fn = function(d){
  x = apply(d, 2, as.numeric) # convert each column to numeric (forced, so NAs result)
  i = apply(x, 2, function(x) sum(x, na.rm=T)) # take the sum of the numeric columns
  names(d)[i != 0] # return only the columns that have a non-zero sum (i.e. they have numeric values)
}

# nc = numeric candidates
nc.ai = numeric.candidates.fn(ai[,-which(names(ai) %in% c("id","iid"))])
nc.ah = numeric.candidates.fn(ah[,-which(names(ah) %in% c("id","iid"))])
nc.bi = numeric.candidates.fn(bi[,-which(names(bi) %in% c("id","iid"))])
nc.bh = numeric.candidates.fn(bh[,-which(names(bh) %in% c("id","iid"))])
nc.ci = numeric.candidates.fn(ci[,-which(names(ci) %in% c("id","iid"))])
nc.ch = numeric.candidates.fn(ch[,-which(names(ch) %in% c("id","iid"))])

# for each numeric candidate, count the number of unique values
# if too few, then even though they are numeric values should treat them as discrete not continuous

lvls1.ai = apply(ai[,nc.ai], 2, function(x) length(levels(as.factor(x))))
lvls1.ah = apply(ah[,nc.ah], 2, function(x) length(levels(as.factor(x))))
lvls1.bi = apply(bi[,nc.bi], 2, function(x) length(levels(as.factor(x))))
lvls1.bh = apply(bh[,nc.bh], 2, function(x) length(levels(as.factor(x))))
lvls1.ci = apply(ci[,nc.ci], 2, function(x) length(levels(as.factor(x))))
lvls1.ch = apply(ch[,nc.ch], 2, function(x) length(levels(as.factor(x))))

# combine the list of all numeric candidate variables

nc.all = data.frame(variable = c(nc.ai, nc.ah, nc.bi, nc.bh, nc.ci, nc.ch),
                    source = c( rep("ai",length(nc.ai)), 
                                rep("ah",length(nc.ah)), 
                                rep("bi",length(nc.bi)), 
                                rep("bh",length(nc.bh)), 
                                rep("ci",length(nc.ci)),
                                rep("ch",length(nc.ch)) ),
                    levels = c(lvls1.ai, lvls1.ah, lvls1.bi, lvls1.bh, lvls1.ci, lvls1.ch) )

# keep only the ones with more than 10 levels

nc.final = nc.all[nc.all$levels > 10, ] 
          
# create dataframes of only the continuous variables
# c.[] = continuous

c.ai = ai[,as.character(nc.final$variable[nc.final$source=='ai'])]
c.ai[is.na(c.ai)] = -1

c.ah = ah[,as.character(nc.final$variable[nc.final$source=='ah'])]
c.ah[is.na(c.ah)] = -1

### ... need to do this ^^ for all countries

########################################################
# Find categorial variables

# find categorical variables that have a manageable number of levels (i.e. throw out ones with, say, 1000 unique values)
# they're all pretty reasonable. Maybe because most values are from a limited-choice questionnaire

nonvars = c("id","iid","poor","country")
lvls2.ai = apply(ai[,-which(names(ai) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))
lvls2.ah = apply(ah[,-which(names(ah) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))
lvls2.bi = apply(bi[,-which(names(bi) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))
lvls2.bh = apply(bh[,-which(names(bh) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))
lvls2.ci = apply(ci[,-which(names(ci) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))
lvls2.ch = apply(ch[,-which(names(ch) %in% c(nc.all$variable, nonvars))], 2, function(x) length(levels(as.factor(x))))

cc.all = data.frame( variable = c(names(lvls2.ai), names(lvls2.ah), names(lvls2.bi), names(lvls2.bh), names(lvls2.ci), names(lvls2.ch)),
                     source = c( rep("ai",length(lvls2.ai)), 
                                 rep("ah",length(lvls2.ah)), 
                                 rep("bi",length(lvls2.bi)), 
                                 rep("bh",length(lvls2.bh)), 
                                 rep("ci",length(lvls2.ci)),
                                 rep("ch",length(lvls2.ch)) ),
                     levels = c(lvls2.ai, lvls2.ah, lvls2.bi, lvls2.bh, lvls2.ci, lvls2.ch))

# get rid of the ones with more than X levels (and they have to have at least 2)

cc.final = cc.all[cc.all$levels <= 10 & cc.all$levels > 1, ] 

# one-hot encoding of the categorical variables
# formulas (f), text (t), variables (v)

f.ai = paste( cc.final$variable[cc.final$source=='ai'], collapse="+")
t.ai = paste0( "v.ai = model.matrix(~1+",f.ai,",ai)[,-1]" )
eval(parse(text=t.ai))

f.ah = paste( cc.final$variable[cc.final$source=='ah'], collapse="+")
t.ah = paste0( "v.ah = model.matrix(~1+",f.ah,",ah)[,-1]" )
eval(parse(text=t.ah))

# f.bi = paste( cc.final$variable[cc.final$source=='bi'], collapse="+")
# t.bi = paste0( "v.bi = model.matrix(~1+",f.bi,",bi)[,-1]" )
# eval(parse(text=t.bi))
# 
# f.bh = paste( cc.final$variable[cc.final$source=='bh'], collapse="+")
# t.bh = paste0( "v.bh = model.matrix(~1+",f.bh,",bh)[,-1]" )
# eval(parse(text=t.bh))

# ...

########################################################
# Create Modeling Dataset


# Create modeling data sets ("m")
# continuous + discretes + outcome variables

mai = data.frame(id = ai$id, uid = paste0(ai$id,"-",ai$iid), poor = ai$poor*1, c.ai, v.ai) # individual data set
mah = data.frame(id = ah$id, poor = ah$poor*1, c.ah, v.ah)                                 # household data set


########################################################
# Build a model on the individual set

# split into 5 random sets (for 5-fold CV)

folds = 5
set.seed(1234)
grps = split(mai$uid, sample(1:folds, nrow(mai), replace=T))

# initialize storage

phat.ai  = list(NULL)
model.ai = list(NULL)

for (i in 1:folds){ # for each random set 
  
  
  # test = sample of size 1/folds
  # train = sample of size (folds-1) / folds
  # e.g. when folds = 5, train on 4/5 of the data and test on 1/5 of the data
  
  test = mai[ mai$uid %in% grps[[i]], ]
  train = mai[ mai$uid %in% unlist(grps[-i]), ]
  
  # Xgboost objections. Remove first 3 columns (id, poor, country) from the predictor variables
  
  train.xgb = xgb.DMatrix( data = data.matrix(train[,-c(1:3)])*1.0, label = data.matrix(train$poor) )
  test.xgb = xgb.DMatrix( data = data.matrix(test[,-c(1:3)])*1.0, label = data.matrix(test$poor) )
  watchlist = list(train=train.xgb, test=test.xgb)
  
  # fit a model (with early stopping on nrounds)
  
  xgb = xgb.train(data = train.xgb, 
                  nrounds = 1000,
                  watchlist = watchlist, 
                  objective = 'binary:logistic', 
                  print_every_n = 5,
                  #verbose = 0,
                  early_stopping_rounds = 15, 
                  eval_metric = 'logloss')
  
  # predict back on test set, and store
  
  probs = predict( xgb, data.matrix(test[,-c(1:3)])*1.0 )
  model.ai[[i]] = xgb
  phat.ai[[i]] = probs
  
}

# derive feature importance

imp.ai = lapply(model.ai, function(x) xgb.importance(names(mai)[-c(1:3)], model=x))
imp.ai = data.frame( variable = unlist(sapply(imp.ai, '[[', 1)), gain = unlist(sapply(imp.ai, '[[', 2)) )
imp.ai = imp.ai %>% group_by(variable) %>% summarize(val = mean(gain)) %>% arrange(desc(val))

# take the 100 most predictive variables at the individual level
# aggregate them up to their mean at the household level

ai.agg = aggregate(mai[,names(mai) %in% imp.ai$variable[1:100]], by=list(mai$id), 'mean')
names(ai.agg)[1] = "id"


########################################################
# Build a model on the household set

# First, add the individual-level aggregate variables into the household dataset

mah = merge(mah, ai.agg, by="id")

# second, repeat the above XGBOOST method

folds = 5
set.seed(5678)
grps = split(mah$id, sample(1:folds, nrow(mah), replace=T))

phat.ah  = list(NULL)
model.ah = list(NULL)

for (i in 1:folds){
  
  test = mah[ mah$id %in% grps[[i]], ]
  train = mah[ mah$id %in% unlist(grps[-i]), ]
  
  train.xgb = xgb.DMatrix( data = data.matrix(train[,-c(1:2)])*1.0, label = data.matrix(train$poor) )
  test.xgb = xgb.DMatrix( data = data.matrix(test[,-c(1:2)])*1.0, label = data.matrix(test$poor) )
  watchlist = list(train=train.xgb, test=test.xgb)
  
  # Parameters (e.g. max_depth) have been set through wimpy trial-and-error
  
  xgb = xgb.train(data = train.xgb, 
                  eta = 0.3,
                  nrounds = 1000,
                  watchlist = watchlist, 
                  objective = 'binary:logistic', 
                  print_every_n = 5,
                  #verbose = 0,
                  early_stopping_rounds = 15, 
                  eval_metric = 'logloss',
                  gamma = 4,
                  max_depth = 2)
  
  probs = predict( xgb, data.matrix(test[,-c(1:2)])*1.0 )
  model.ah[[i]] = xgb
  phat.ah[[i]] = probs
  
}


########################################################
# Calculate expected Log-Loss

# Collect out-of-sample predictions on the full dataset

ids = unlist(grps)
preds = unlist(phat.ah)
df = data.frame(id=ids, phat=preds)
df = merge(ah[,c("id","poor")], df, by="id")

# look at distributions

plot(density(df$phat[df$poor==TRUE]))
lines(density(df$phat[df$poor==FALSE]), col='blue')

# Log-loss
# y*log(yhat) + (1-y)*log(1-yhat)

lli = -(df$poor*1*log(df$phat) + (1-df$poor*1)*log(1-df$phat))
mean(lli)

# Variable importance

imp.ah = lapply(model.ah, function(x) xgb.importance(names(mah)[-c(1:2)], model=x))
imp.ah = data.frame( variable = unlist(sapply(imp.ah, '[[', 1)), gain = unlist(sapply(imp.ah, '[[', 2)) )
imp.ah = imp.ah %>% group_by(variable) %>% summarize(val = mean(gain)) %>% arrange(desc(val))
View(imp.ah)




