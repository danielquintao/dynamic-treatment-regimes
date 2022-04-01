# DATA GENERATION ADAPTED FROM DOCUMENTATION OF DTRreg PACKAGE
# https://cran.r-project.org/web/packages/DTRreg/DTRreg.pdf (pag 15)

n = 120  # will actually build 2x more
psi.mat<-matrix(rep(c(1,1,-1),2),byrow=TRUE,nrow=2)
alpha.mat<-matrix(rep(c(-1,1),2),byrow=TRUE,nrow=2)

simu <- function(){
  # stage 1
  x1<-abs(rnorm(n,0,1))
  a1<-sign(rnorm(n,alpha.mat[1,1]+alpha.mat[1,2]*x1,1))
  # stage 2
  x2<-abs(rnorm(n,0,1))
  a2<-sign(rnorm(n,alpha.mat[2,1]+alpha.mat[2,2]*x2,1))
  # blips
  gamma1<-as.matrix(cbind(a1,a1*x1))
  gamma2<-as.matrix(cbind(a2,a2*x2))
  # y: outcome
  # y <- trmt free + blip
  y<-log(x1)+sin(x1)+log(x2)+sin(x2)+gamma1+gamma2 + rnorm(n,0,1)
  # convert to a vector for formatting into data frame
  y <- as.vector(y)
  # EDIT I'll add a column of zeros to be compatible with my code
  y1 <- rep(0, length(y))
  # a to category
  a1 <- replace(a1, a1==1, "A")
  a1 <- replace(a1, a1==-1, "B")
  a2 <- replace(a2, a2==1, "A")
  a2 <- replace(a2, a2==-1, "B")
  # data
  data<-data.frame(cbind(y1,y,x1,x2,a1,a2))
}

for (i in c(1:50)) {
  data <- simu()
  write.csv(data, paste(c("./data/simulation",i,".csv"),collapse=""))
}
