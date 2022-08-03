##########################################################
# Create edx set, validation set (final hold-out test set) # It takes a few minutes...
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
# download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
# download.file("https://owncloud.univ-artois.fr/index.php/s/3Oj588P4tiG992t/download", dl) 
#  alternative download links

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Loss function
##########################################################

RMSE <- function(true_ratings, predicted_ratings)
{ sqrt(mean((true_ratings - predicted_ratings)^2))
}
  
##########################################################
# A first simple model
##########################################################

#we assume that the same rating *mu* is given for all movies and users 
mu <- mean(edx$rating) 
mu
new_rmse <- RMSE(validation$rating, mu) 
new_rmse

#creating a results table:
rmse_results <- tibble(method = "Just the average", RMSE = new_rmse)
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()

##########################################################
# A second model with movie effects
##########################################################

#We can make the hypothesis that some films are better rated than others by all the spectators
# b_i represent this movie effects
mu=mean(edx$rating)
bi <- edx %>% 
  group_by(movieId) %>% 
  summarize(bi = mean(rating - mu))

# the distribution of bi
ggplot(bi)+aes(x=bi)+geom_histogram(bins = 40)

# We construct predictors with movie effects 
prediction_test <- validation %>% 
  left_join(bi, by = "movieId") %>% 
  mutate(rating_hat = (mu + bi))

# We calculate the new RMSE:
new_rmse    <- RMSE(prediction_test$rating_hat,validation$rating)
new_rmse

# previous table completed
rmse_results <- bind_rows(rmse_results,
                data_frame(method="with movie effects",  
                RMSE = new_rmse))
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()


##########################################################
# A third model with user effects
##########################################################

#We can also make the assumption that there is significant variability among users
# b_u represent this users effects
bu <- edx %>% 
      left_join(bi, by = "movieId") %>% 
      group_by(userId) %>% 
      summarize(bu = mean(rating -bi - mu))

# the distribution of bu
ggplot(bu)+aes(x=bu)+geom_histogram(bins = 40)

# We construct predictors with movie effects 
prediction_test <- validation %>% 
                   left_join(bi, by = "movieId") %>% 
                   left_join(bu, by = "userId") %>% 
                   mutate(rating_hat = (mu + bu + bi))
# We calculate the new RMSE:
new_rmse <- RMSE(prediction_test$rating_hat,validation$rating)
new_rmse

# previous table completed
rmse_results <- bind_rows(rmse_results,
                data_frame(method="with movie and user effects",  
                RMSE = new_rmse))
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()


##########################################################
# A fourth  model with genres effects
##########################################################

# we will now take into account the genres of the films
# b_g represent this genres effects  
bg <- edx %>% 
  left_join(bi, by = "movieId") %>% 
  left_join(bu, by = "userId") %>% 
  group_by(genres) %>% 
  summarize(bg = mean(rating -bi -bu - mu))

# the distribution of bg
ggplot(bg)+aes(x=bg)+geom_histogram(bins = 40)

# We construct predictors with movie effects 
prediction_test <- validation %>% 
                   left_join(bi, by = "movieId") %>% 
                   left_join(bu, by = "userId") %>% 
                   left_join(bg, by = "genres") %>%  
                   mutate(rating_hat = (mu + bu + bi + bg))

# We calculate the new RMSE:
new_rmse <- RMSE(prediction_test$rating_hat,validation$rating)
new_rmse

# previous table completed
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="with movie, user and genres effects",  
                                     RMSE = new_rmse))
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()


##########################################################
# A fifth  model with a times stamp effects
##########################################################
# We will now take into account a times stamp effect on the  films ratting. 
# b_t represent this times effects
# The average will be done here on each of the different weeks of the times stamp. Using the **as_datetime** function that converts the data in the **timestamps** column to a date and the **round_date** function that determines the week number of this date.
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)
bt <- edx %>% 
      mutate(weeks = round_date(as_datetime(timestamp),"week")) %>% 
      left_join(bi, by = "movieId") %>% 
      left_join(bu, by = "userId") %>% 
      left_join(bg, by = "genres") %>% 
      group_by(weeks) %>% 
      summarize(bt = mean(rating -bi -bu - bg - mu))

# the distribution of bt
ggplot(bt)+aes(x=bt)+geom_histogram(bins = 40)

# We construct predictors with movie effects 
prediction_test <- validation %>% 
                   mutate(weeks = round_date(as_datetime(timestamp),"week")) %>%
                   left_join(bi, by = "movieId") %>% 
                   left_join(bu, by = "userId") %>% 
                   left_join(bg, by = "genres") %>%  
                   left_join(bt, by = "weeks") %>%  
                   mutate(rating_hat = (mu + bu + bi + bg + bt))

# We calculate the new RMSE:
new_rmse <- RMSE(prediction_test$rating_hat,validation$rating)
new_rmse

# previous table completed
rmse_results <- bind_rows(rmse_results,
                data_frame(method="with movie, user, genres and times effects",  
                RMSE = new_rmse))
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()


##########################################################
# Regularization (It takes a few minutes... But less than 10 minutes)
##########################################################

## Averages with penalty coefficients and choice of the best penalty coefficient
lambdas <- seq(1, 9, 0.5)
rmses <- sapply(lambdas, function(l){
  mu=mean(edx$rating)
  
  bi <- edx %>% 
    group_by(movieId) %>% 
    summarize(bi = sum(rating - mu)/(n()+l))
  
  bu <- edx %>% 
    left_join(bi, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(bu = sum(rating -bi - mu)/(n()+l))
  
  bg <- edx %>% 
    left_join(bi, by = "movieId") %>% 
    left_join(bu, by = "userId") %>% 
    group_by(genres) %>% 
    summarize(bg = sum(rating -bi -bu - mu)/(n()+l))
  
  bt <- edx %>% 
    mutate(weeks = round_date(as_datetime(timestamp),"week")) %>%
    left_join(bi, by = "movieId") %>% 
    left_join(bu, by = "userId") %>% 
    left_join(bg, by = "genres") %>% 
    group_by(weeks) %>% 
    summarize(bt = sum(rating -bi -bu - bg - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    mutate(weeks = round_date(as_datetime(timestamp),"week")) %>%
    left_join(bi, by = "movieId") %>% 
    left_join(bu, by = "userId") %>% 
    left_join(bg, by = "genres") %>%  
    left_join(bt, by = "weeks") %>%  
    mutate(rating_hat = (mu + bu + bi + bg + bt)) 
  
  return(RMSE(predicted_ratings$rating_hat, validation$rating))
})


# RMSE as function of lambda
qplot(lambdas, rmses)

# lambda that minimizes the RMSE is then
lambdas[which.min(rmses)]

# associated RMSE  
new_rmse <- min(rmses)

# previous table completed
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="with all effects and penalty coefficient",  
                                     RMSE = new_rmse))
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
library(knitr)
as_tibble(rmse_results)  %>% kable()

