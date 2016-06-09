# Set working directory
setwd('C:/Users/xu labo/Downloads/scripts/kaggle/Facial Keypoints Detection')

# Prepare the data
# dir and files
data.dir <- 'data/'
train.file <- paste0(data.dir, 'training.csv')
test.file <- paste0(data.dir, 'test.csv')

# load data
# d.train <- read.csv(train.file, stringsAsFactors=F)
# im.train <- d.train$Image
# d.train$Image <- NULL
# 
# library(foreach)
# im.train <- foreach(im = im.train, .combine=rbind) %do% {
#     as.integer(unlist(strsplit(im, " ")))
# }
# 
# d.test  <- read.csv(test.file, stringsAsFactors=F)
# im.test <- foreach(im = d.test$Image, .combine=rbind) %do% {
#     as.integer(unlist(strsplit(im, " ")))
# }
# d.test$Image <- NULL
# 
# save(d.train, im.train, d.test, im.test, file='data.Rd')

# If data has been saved, just load it
load('data.Rd')

# Example visualization
imVis <- function() {
    im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96)
    image(1:96, 1:96, im, col=gray((0:255)/255))
    points(96-d.train$nose_tip_x[1],         96-d.train$nose_tip_y[1],         col="red")
    points(96-d.train$left_eye_center_x[1],  96-d.train$left_eye_center_y[1],  col="blue")
    points(96-d.train$right_eye_center_x[1], 96-d.train$right_eye_center_y[1], col="green")
}

variaVis <- function() {
    im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96)
    for(i in 1:nrow(d.train)) {
        points(96-d.train$nose_tip_x[i], 96-d.train$nose_tip_y[i], col="red")
    }
}

imMaxVis <- function() {
    idx <- which.max(d.train$nose_tip_x)
    imMax <- matrix(data=rev(im.train[idx,]), nrow=96, ncol=96)
    image(1:96, 1:96, imMax, col=gray((0:255)/255))
    points(96-d.train$nose_tip_x[idx], 96-d.train$nose_tip_y[idx], col="red")
}

# imVis()
# variaVis()
# imMaxVis()

# First submission
firstSub <- function() {
    p           <- matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)
    colnames(p) <- names(d.train)
    predictions <- data.frame(ImageId = 1:nrow(d.test), p)
    head(predictions)
    
    library(reshape2)
    submission <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
    head(submission)
    
    example.submission <- read.csv(paste0(data.dir, 'IdLookupTable.csv'))
    sub.col.names      <- names(example.submission)
    example.submission$Location <- NULL
    submission <- merge(example.submission, submission, all.x=T, sort=F)
    submission <- submission[, sub.col.names]
    
    submission$ImageId <- NULL
    submission$FeatureName <- NULL
    
    write.csv(submission, file="submission_means.csv", quote=F, row.names=F)
}

coord      <- "left_eye_center"
patch_size <- 10

coord_x <- paste(coord, "x", sep="_")
coord_y <- paste(coord, "y", sep="_")
library(foreach)
patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %do% {
    im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
    x   <- d.train[i, coord_x]
    y   <- d.train[i, coord_y]
    x1  <- (x-patch_size)
    x2  <- (x+patch_size)
    y1  <- (y-patch_size)
    y2  <- (y+patch_size)
    if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
    {
        as.vector(im[x1:x2, y1:y2])
    }
    else
    {
        NULL
    }
}
mean.patch <- matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
# image(1:21, 1:21, mean.patch[21:1,21:1], col=gray((0:255)/255))

search_size <- 2
mean_x <- mean(d.train[, coord_x], na.rm=T)
mean_y <- mean(d.train[, coord_y], na.rm=T)
x1     <- as.integer(mean_x)-search_size
x2     <- as.integer(mean_x)+search_size
y1     <- as.integer(mean_y)-search_size
y2     <- as.integer(mean_y)+search_size
params <- expand.grid(x = x1:x2, y = y1:y2)

im <- matrix(data = im.test[1,], nrow=96, ncol=96)

r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
    x     <- params$x[j]
    y     <- params$y[j]
    p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
    score <- cor(as.vector(p), as.vector(mean.patch))
    score <- ifelse(is.na(score), 0, score)
    data.frame(x, y, score)
}

best <- r[which.max(r$score), c("x", "y")]



