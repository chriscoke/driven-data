
########################################################################
## Overview

# See this page for more information -- https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

# This is a work in progress. Doing some initial exploration and model building, and seeing where this predictive capacity thus far
# rougly falls in the range of submissions to gauge my progress.



########################################################################
## Libraries

library(data.table)
library(nnet)
library(glmnet)
library(randomForest)
library(dplyr)
library(ggplot2)
library(knitr)

########################################################################
## Import Data

setwd("your_folder_here")
setwd("~/../Documents/DrivenData/")

train1 = fread("train_vals.csv")
train2 = fread("train_labs.csv")
train = merge(train1, train2, by="id")
rm(train1, train2)
test = fread("test_vals.csv")

# remove some variables to start with
# these are either pure duplicates, unused variables (for now), or discrete variables that have a more granular equivalent

sub = data.frame(train[,-c("latitude","longitude","date_recorded",
                           "region_code", "quantity_group", "source_type","waterpoint_type_group",
                           "quality_group","payment_type","extraction_type_group", "scheme_management",
                           "extraction_type_class","source_class","management_group")])


########################################################################
## Data Exploration

agg = sub %>% group_by(status_group) %>% summarize(count = n(), percent = n()/nrow(sub))
kable(agg)

p1 = ggplot(sub, aes(x=status_group, fill=status_group))
p2 = geom_bar()
p3 = guides(fill=FALSE)
p4 = labs(title = "Water Pump Statuses")
p1+p2+p3+p4

# ... other descriptive / exploratory stuff redacted. Just getting a sense of the data here.


########################################################################
## Impute data
## very simple structure right now: if it's missing a value, fill it in with "unknown"

bool = apply(sub, 2, function(x) is.na(x) | x=="")
sub[bool] = "unknown"


########################################################################
## Variable transformation and removal

# Generate discrete variables out of common value continuous IVs
sub["zero_tsh"] = 0; sub[which(sub$amount_tsh==0),"zero_tsh"] = 1
sub["zero_pop"] = 0; sub[which(sub$population==0),"zero_pop"] = 1
sub["zero_priv"] = 0; sub[which(sub$num_private==0),"zero_priv"] = 1

# treat these integers as factors
sub$num_private = as.factor(sub$num_private)
sub$district_code = as.factor(sub$district_code)

# convert the construction year into an age group
sub["age"] = 2016 - sub$construction_year
brks = hist(sub$age[sub$age!=2016], plot=F)$breaks
assign = cut(sub$age, brks, include.lowest=T, right=F, labels=F)
assign[is.na(assign)] = 999
sub["age_grp"] = as.factor(assign)
sub = sub[,-which(names(sub) %in% c("age","construction_year"))]

# remove variables with too many factor levels
# with only ~50k records, not likely to get too much out of these without some transformation
# takes out: funder, installer, wpt_name, subvillage, lga, ward, scheme_name, recorded_by
classes = sapply(sub, 'class')
fact.cols = which(classes %in% c("character","factor")==T)
lvls = apply(sub[,fact.cols], 2, function(x) length(levels(as.factor(x))))
rem = names(sub[,fact.cols])[which(lvls > 25 | lvls <= 1)]
sub = sub[,-which(names(sub) %in% rem)]

########################################################################
## Generate dataset components: outcome variable, continuous, discrete, interactions, polynomials

classes = sapply(sub, 'class')

# dependent variable
y = sub$status_group

# identifiers
ids = sub$id

# continuous (or already indicator) independent variables 
cont.cols = which(classes %in% c("character","factor")==F)
cont = sub[,cont.cols]
cont = cont[,-1]

# discrete dependent variables: convert into 0/1 indicators
disc.names = names(classes[classes %in% c("character","factor")])
disc.names = disc.names[-1]
txt.disc = paste0("disc = model.matrix(y ~ ",paste(disc.names, collapse="+"),",data=sub)[,-1]")
eval(parse(text=txt.disc))

# model data
mdta = cbind(ids, y, cont, disc)
names(mdta) = gsub("-| |/",".", names(mdta))

# interaction terms ## NOT INCLUDING THESE YET
# inter.names = names(cont)
# txt.inter = paste0("inters = model.matrix(~(",paste(names(mdta), collapse="+"),")^2, mdta, na.action='na.pass')")
# eval(parse(text=txt.inter))
# inter = inters[,grep(":",colnames(inters))] #keep only interaction terms


########################################################################
## Separate into test and training sets
# actually here, I'm technically splitting the official training set into two sets

set.seed(1234)
smpl = sample(1:nrow(mdta), nrow(mdta)/2, replace=F)
traind = mdta[smpl,]
testd = mdta[-smpl,]

########################################################################
## Fit some models

# Multinomial Logit model

start = Sys.time()
m1 = multinom(y ~ ., data=traind)             ## WARNING: THIS TAKES ABOUT 60 SECONDS TO RUN
end = Sys.time()
end - start

pred = predict(m1, newdata = testd, type="class")
act = testd$y
table(pred, act)
missclass = length(which(pred != act))/length(act)
m1.classrate = 1-missclass

# Same thing, but using cross-validation and lasso shrinkage method to reduce variables

# x = as.matrix(mdta[,-c(1:2)])
# y = as.matrix(mdta[,2])
# start = Sys.time()
# m2 = cv.glmnet(x, y, family="multinomial", type.multinomial = "grouped")  # WARNING: THIS TAKES ABOUT 1 HOUR TO RUN
# end = Sys.time()
# end - start
# plot(m2)
# pred = predict(m2, newx = x, type="class", s=m2$lambda.1se)
# table(pred,y)
# missclass = length(which(pred != y))/length(y)
# m2.classrate = 1-missclass
# 
# coefs = coef(m2, s=m2$lambda.1se) #coefficients of model with the cv error rate w/in 1 s.d. of the minimum error model

# Random Forest

x = as.matrix(traind[,-c(1:2)])
y = as.matrix(traind[,2])

start = Sys.time()
m3 = randomForest(as.factor(y)~., data=data.frame(y,x), importance=TRUE, ntree=100) # WARNING: THIS TAKES ABOUT 10 MINUTES TO RUN
end = Sys.time()
end - start
pred = predict(m3, newdata = testd, type="class")
act = testd$y
table(pred,act)
missclass = length(which(pred != act))/length(act)
m3.classrate = 1-missclass


########################################################################
## Connect to webpage and get table of submissions to compare


url = "https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/leaderboard/"
r = readLines(url)
find = grep("values=",r)
tbl = r[find]

clean.fn = function(x){
  rem = '<td align=|center|><span class=|inlinesparkline|values=|></span></td>|"| '
  xn = gsub(rem,"",x)  
  clean = as.numeric(unlist(strsplit(xn,",")))
}

tbl.vals = lapply(tbl, function(x) clean.fn(x))
best = sapply(tbl.vals, 'max')
plot(density(best),xlim=c(0.5,0.9), main="Classification Rates submitted")
abline(v=c(m1.classrate, m3.classrate),col=c("blue","red"),lwd=2)
legend("topleft", lty=c(1,1,1), col=c("black","blue","red"), c("submissions", "me: logistic", "me: random forest"), cex=0.75)




############### Variables

# 
# amount_tsh - Total static head (amount water available to waterpoint)
# date_recorded - The date the row was entered
# funder - Who funded the well
# gps_height - Altitude of the well
# installer - Organization that installed the well
# longitude - GPS coordinate
# latitude - GPS coordinate
# wpt_name - Name of the waterpoint if there is one
# basin - Geographic water basin
# subvillage - Geographic location
# region - Geographic location
# region_code - Geographic location (coded)
# district_code - Geographic location (coded)
# lga - Geographic location
# ward - Geographic location
# population - Population around the well
# public_meeting - True/False
# recorded_by - Group entering this row of data
# scheme_management - Who operates the waterpoint
# scheme_name - Who operates the waterpoint
# permit - If the waterpoint is permitted
# construction_year
# extraction_type
# extraction_type_group
# extraction_type_class
# management
# management_group
# payment
# payment_type
# quality_group
# quantity
# source
# source_class
# waterpoint_type





