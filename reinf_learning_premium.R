install.packages(c("xts","vars","rugarch","urca","fBasics","xtable","egcm"))

library(xts)
library(urca)
library(vars)
library(fBasics)
library(xtable)
library(rugarch)
library(egcm)


dane<-list()
dane[[1]]<-read.csv2("cds1.csv",header=TRUE)
dane[[2]]<-read.csv("2016-2018.csv",header=TRUE)

for(i in 1:length(dane))
{
  dane[[i]]<-xts(dane[[i]][,-1],order.by=as.Date(dane[[i]][,1]))
}
dane<-Reduce(merge,dane)

dane2<-xts(merge(rowMeans(dane[,1:2],na.rm=TRUE)/dane[,3],dane[,4]),order.by=index(dane))
dane3<-na.omit(na.locf(dane2["2016/2018"]))
dane<-xts(merge(dane3[,1],cumprod(dane3[,2])),order.by = index(dane3))

stopy.zwrotu<-na.omit(merge(diff(log(dane3[,1])),dane3[,2]-1))

dane.calosc<-merge(dane,stopy.zwrotu)

colnames(dane.calosc)<-c("CDS.prem5Y","Portfel","CDS.ret","Portfel.ret")
opisowe<-basicStats(dane.calosc)

par(mfrow=c(2,2))
for(i in 1:ncol(dane.calosc))
{
  print(plot(dane.calosc[,i],main=colnames(dane.calosc[,i]),major.ticks="quarters",grid.ticks.on="quarters"))
}

xtable(opisowe,digits=4)

egcm(dane.calosc[,2],dane.calosc[,1])
summary(egcm(dane.calosc[,2],dane.calosc[,1]))


summary(ca.jo(dane.calosc[,1:2],ecdet="none",spec="longrun"))

summary(lm(dane.calosc[,3]~dane.calosc[,4]))


specyfikacja<-ugarchspec(mean.model=list(armaOrder=c(1,0),include.mean=TRUE,archm=TRUE,archpow=2,
                                         external.regressors=stopy.zwrotu[,1]),
                         distribution.model="sstd",
                         variance.model=c(list(model="sGARCH",garchOrder=c(1,1))))

model<-ugarchfit(specyfikacja,stopy.zwrotu[,2])
model
xtable(model@fit$matcoef,digits=4)

colnames(stopy.zwrotu)<-c("CDS","Portfel")
model1<-VAR(stopy.zwrotu,p=2,type="const")
summary(model1)
causality(model1,cause="CDS")
causality(model1,cause="Portfel")
