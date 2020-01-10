if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(ggthemes)
library(tidyverse)
library(caret)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

#Extracting Year From Title Of Movie
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))


#Splitting Edx Dataset And Validation Dataset Using seperate_rows()
split_edx <- edx %>% separate_rows(genres,sep = "\\|")
split_validation <- validation %>% separate_rows(genres,sep = "\\|")


#Exploratory Data Analysis
#Total Unique Users And Movies In Data
edx %>% summarise(total_users=n_distinct(userId),total_movies=n_distinct(movieId)) %>% 
  knitr::kable()

#Movies Produced In Every Year
edx %>% group_by(year) %>% summarise(total_movies = n()) %>% ggplot(aes(year,total_movies)) + geom_bar(stat = "identity") + theme_gdocs() + xlab("Year") + ylab("Total Movies Produced") + ggtitle("Distribution Of Total Movies Produced Every Year")

#Movies Produced By Genre
split_edx %>% filter(genres != "(no genres listed)") %>% group_by(genres) %>% 
  summarise(total_movies = n()) %>% 
  ggplot(aes(reorder(genres,-total_movies),total_movies)) + geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90,hjust = 1, vjust = 0)) + xlab("Genres") + ylab("Total Movies Produced") + ggtitle("Distribution Of Total Movies Produced In Every Genre")
#Number Of Active Users Over The Years

#Exploring The Trend Of The Average Ratings Over The Years (Release Years)
edx %>% group_by(year) %>% summarise(mean_rating = mean(rating)) %>% ggplot(aes(year,mean_rating)) + geom_point(size = 3) + geom_smooth() + ggtitle("Average Ratings Versus Release Year") + xlab("Release Year") + ylab("Average Rating") + theme_gdocs()

#Number Of Ratings In Different Genres
split_edx %>% filter(genres != "(no genres listed)") %>% group_by(genres) %>% summarise(total_ratings = n()) %>% knitr::kable()
split_edx %>% filter(genres != "(no genres listed)") %>% group_by(genres) %>% summarise(total_ratings = n()) %>% ggplot(aes(reorder(genres,-total_ratings),total_ratings)) + geom_bar(stat = "identity") + theme(axis.text.x=element_text(angle=90,hjust=1, vjust=0)) + ylab("Number Of Ratings") + xlab("Genres") + ggtitle("Number Of Ratings Given To Different Genres")

#Top 5 Movie Genres Rated The Most Number Of Times
split_edx %>% group_by(genres) %>% summarise(total_ratings = n()) %>% arrange(desc(total_ratings)) %>% head(5) %>% knitr::kable()

#Exploring The Trend Of The Average Ratings Over The Years For Top 5 Most Rated Movie Genres
split_edx %>% filter(genres %in% c("Drama","Comedy","Action","Thriller","Adventure")) %>% group_by(year,genres) %>% summarise(mean_rating = mean(rating)) %>% ggplot(aes(year,mean_rating,color = genres)) + geom_line() + ggtitle("Rating Trend For Top 5 Most Rated Genres") + xlab("Release Year") + ylab("Average Rating") + theme_gdocs()

#Top 10 Most Rated Movies
edx %>% group_by(title) %>% summarise(total_ratings = n()) %>% arrange(desc(total_ratings)) %>% head(10) %>% knitr::kable()

#Top 10 Highest Rated (Average Rating) Movie
edx %>% group_by(title) %>% summarise(average_rating = mean(rating)) %>% arrange(desc(average_rating)) %>% head(10)

#Count Of Number Of Ratings Given To Different Movies
edx %>% count(movieId) %>% ggplot(aes(n)) + geom_histogram(bins = 30,color = "black") + scale_x_log10() + xlab("Number Of Ratings Given") + ylab("Count Of Movies") + theme_gdocs()

#Count Occurences Of Different Ratings Given By Users
edx %>% group_by(rating) %>% summarise(total_ratings = n()) %>% ggplot(aes(rating,total_ratings)) + geom_bar(stat = "identity") + xlab("Ratings") + ylab("Number Of Ratings") + ggtitle("Count Of Occurences Of Different Ratings Given By Users") + theme_gdocs()

#Count Of Number Of Ratings Given By Different Users
edx %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(bins = 30,color = "black") + scale_x_log10() + xlab("Users") + ylab("Number Of Ratings Given") + theme_gdocs()

#Recommendation Models

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2,na.rm=T))
}

rmse_results <- data.frame()

#Basic Model
mu <- mean(edx$rating)
rmse <- RMSE(validation$rating,mu)

rmse_results <- data.frame(Method = "Using Mean Only",RMSE = rmse)
rmse_results %>% knitr::kable()

#Model Using Movie Effect
movie_avg <- edx %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu))
movie_avg %>% ggplot(aes(b_i)) + geom_histogram(bins = 20,color = "black") + ggtitle("Penalty Term b_i (Movie Effect)") + theme_gdocs() 

movie_avg_pred <- validation %>% left_join(movie_avg,by = "movieId") %>% mutate(pred = (mu + b_i)) %>% pull(pred)
rmse <- RMSE(validation$rating,movie_avg_pred)

rmse_results <- bind_rows(rmse_results,data.frame(Method = "Using Movie Effect",RMSE = rmse))
rmse_results %>% knitr::kable()

#Model Using User Effect
user_avg <- edx %>% left_join(movie_avg,by = "movieId") %>% group_by(userId) %>% summarise(b_u = mean(rating - mu - b_i))
user_avg %>% ggplot(aes(b_u)) + geom_histogram(bins = 30,color = "black") + ggtitle("Penalty Term b_u (User Effect)") +theme_gdocs()

user_avg_pred <- validation %>% left_join(movie_avg,by = "movieId") %>% left_join(user_avg,by = "userId") %>% mutate(pred = b_u + b_i + mu) %>% pull(pred)
rmse <- RMSE(validation$rating,user_avg_pred)

rmse_results <- bind_rows(rmse_results,data.frame(Method = "Using Movie & User Effect",RMSE = rmse))
rmse_results %>% knitr::kable()

#Model Using Year Effect
year_avg <- edx %>% left_join(movie_avg,by = "movieId") %>% left_join(user_avg,by = "userId") %>% group_by(year) %>% summarise(b_y = mean(rating - mu - b_i - b_u))
year_avg %>% ggplot(aes(b_y)) + geom_histogram(bins = 30,color = "black") + ggtitle("Penalty Term b_y (Year Effect)") + theme_gdocs()

year_avg_pred <- validation %>% left_join(movie_avg,by = "movieId") %>% left_join(user_avg,by = "userId") %>% left_join(year_avg,by = "year") %>% mutate(pred = b_u + b_i + b_y + mu) %>% pull(pred)
rmse <- RMSE(validation$rating,year_avg_pred)

rmse_results <- bind_rows(rmse_results,data.frame(Method = "Using Movie,User & Year Effect",RMSE = rmse))
rmse_results %>% knitr::kable()

#Regularised User & Movie Effect Model
lambdas <- seq(0, 10, 0.25)
rmse_list <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  
  movie_avg <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  user_avg <- edx %>% 
    left_join(movie_avg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  reg_user_movie_pred <- validation %>% 
    left_join(movie_avg, by = "movieId") %>%
    left_join(user_avg, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% pull(pred)
  
  return(RMSE(validation$rating,reg_user_movie_pred))
})

data.frame(lambda = lambdas,rmse = rmse_list) %>% ggplot(aes(lambda,rmse)) + geom_point(size = 3) + theme_gdocs() + xlab("Lambda Values") + ylab("RMSE Values") + theme_gdocs()

lambda <- lambdas[which.min(rmse_list)]

mu <- mean(edx$rating)

movie_avg <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

user_avg <- edx %>% 
  left_join(movie_avg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

reg_user_movie_pred <- validation %>% 
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

rmse <- RMSE(validation$rating,reg_user_movie_pred)

rmse_results <- bind_rows(rmse_results,data.frame(Method = "Regularised Method Using Movie & User Effect",RMSE = rmse))
rmse_results %>% knitr::kable()

#Regularized Model Using Movie, User, Year & Genre Effect
lambdas <- seq(0, 15, 1)
rmse_list <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  
  movie_avg <- split_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  user_avg <- split_edx %>% 
    left_join(movie_avg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  year_avg <- split_edx %>%
    left_join(movie_avg, by='movieId') %>%
    left_join(user_avg, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+l), n_y = n())
  
  genre_avg <- split_edx %>%
    left_join(movie_avg, by='movieId') %>%
    left_join(user_avg, by='userId') %>%
    left_join(year_avg, by = 'year') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+l), n_g = n())
  
  reg_pred <- split_validation %>% 
    left_join(movie_avg, by='movieId') %>%
    left_join(user_avg, by='userId') %>%
    left_join(year_avg, by = 'year') %>%
    left_join(genre_avg, by = 'genres') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>% pull(pred)
  
  return(RMSE(split_validation$rating,reg_pred))
})

data.frame(lambda = lambdas,rmse = rmse_list) %>% ggplot(aes(lambda,rmse)) + geom_point(size = 3) + theme_gdocs() + xlab("Lambda Values") + ylab("RMSE Values")
lambda <- lambdas[which.min(rmse_list)]

mu <- mean(edx$rating)

movie_avg <- split_edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

user_avg <- split_edx %>% 
  left_join(movie_avg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

year_avg <- split_edx %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u)/(n()+lambda), n_y = n())

genre_avg <- split_edx %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(year_avg, by = 'year') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+lambda), n_g = n())

reg_pred <- split_validation %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(year_avg, by = 'year') %>%
  left_join(genre_avg, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>% pull(pred)

rmse <- RMSE(split_validation$rating,reg_pred)

rmse_results <- bind_rows(rmse_results,data.frame(Method = "Regularised Method Using Movie,User,Year & Genre Effect",RMSE = rmse))
rmse_results %>% knitr::kable()