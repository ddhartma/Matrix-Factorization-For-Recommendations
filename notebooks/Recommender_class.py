class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self):
        '''
        what do we need to start out our recommender system
        '''
        pass

    def read_dataset(self, movies_path='./data/movies_clean.csv', reviews_path='./data/train_data.csv'):
        '''
        INPUTS:
        ------------
            movies_path - (string) file path to the movies, default='./movies_clean.csv'
            reviews_path - (string) file path to the reviews, default='./train_data.csv'

        OUTPUTS:
        ------------
            movies - (dataframe) movie dataframe
            reviews - (dataframe) review dataframe
        '''

        # Read in the datasets
        movies = pd.read_csv(movies_path)
        reviews = pd.read_csv(reviews_path)

        del movies['Unnamed: 0']
        del reviews['Unnamed: 0']

        print('movies')
        print(movies.head())
        print(movies.shape)
        print('------------------------')
        print(' ')
        print('reviews')
        print(reviews.head())
        print(reviews.shape)
        print('------------------------')
        print(' ')

        return movies, reviews

    def create_train_test(self,reviews, order_by, train_size_prct=0.8):
        '''
        INPUTS:
        ------------
            reviews - (pandas df) dataframe to split into train and test
            order_by - (string) column name to sort by
            train_size_prct - (float) - percentage of data used for training, default=0.8

        OUTPUTS:
        ------------
            training_df -  (pandas df) dataframe of the training set
            validation_df - (pandas df) dataframe of the test set
        '''

        # Define the train and test data size via train_size_prct
        training_size = int(reviews.shape[0] * train_size_prct)
        testing_size = reviews.shape[0] - training_size

        # Sort the reviews by date before splitting
        # use old data for training, new data for validation
        reviews_new = reviews.sort_values(order_by)
        training_df = reviews_new.head(training_size)
        validation_df = reviews_new.iloc[training_size:training_size+testing_size]

        print('reviews_new')
        print(reviews_new.head())
        print(reviews_new.shape)
        print('------------------------')
        print(' ')
        print('training_df')
        print(training_df.head())
        print(training_df.shape)
        print('------------------------')
        print(' ')
        print('validation_df')
        print(validation_df.head())
        print(validation_df.shape)
        print('------------------------')
        print(' ')


        return training_df, validation_df

    def fit(self,
            movies_path='./data/movies_clean.csv',
            reviews_path='./data/train_data.csv',
            order_by='date',
            train_size_prct=0.8,
            latent_features=15,
            learning_rate=0.005,
            iters=10):

        ''' Fit the recommender engine to the dataset and
            save the results to pull from when you need to make predictions

        INPUTS:
        ------------
            movies_path - (string) file path to the movies, default='./movies_clean.csv'
            reviews_path - (string) file path to the reviews, default='./train_data.csv'
            order_by - (string) column name to sort by
            train_size_prct - (float) - percentage of data used for training, default=0.8
            latent_features - (int) the number of latent features used, default=15,
            learning_rate - (float) the learning rate, default=0.005
            iters - (int) the number of iterations, default=10

        OUTPUTS:
        -------------
            user_mat - (numpy array) a user by latent feature matrix
            movie_mat - (numpy array) a latent feature by movie matrix

        '''
        # Read in movie and review DataFrames
        movies, reviews = self.read_dataset(movies_path,reviews_path)

        # Hyperparameters: Number of latent features, lr, epochs
        latent_features = latent_features
        learning_rate = learning_rate
        iters = iters

        training_df, validation_df = self.create_train_test(reviews, order_by, train_size_prct)

        # Create user-by-item matrix as np array
        train_user_item = training_df[['user_id', 'movie_id', 'rating', 'timestamp']]
        train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        ratings_mat = np.array(train_data_df)
        self.ratings_mat = ratings_mat

        print('user-by-item matrix')
        print(ratings_mat)
        print(ratings_mat.shape)
        print('------------------------')
        print(' ')

        # Number of users and movies in the user-by-item matrix
        self.n_users = ratings_mat.shape[0]
        self.n_movies = ratings_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

        print('number of users: ', self.n_users)
        print('number of movies: ', self.n_movies)
        print('number of non nan ratings: ', self.num_ratings)

        # Initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, latent_features)
        movie_mat = np.random.rand(latent_features, self.n_movies)

        print('U matrix (users) before training')
        print(user_mat)
        print(user_mat.shape)
        print('------------------------')
        print(' ')

        print('Vt matrix (movies) before training')
        print(movie_mat)
        print(movie_mat.shape)
        print('------------------------')
        print(' ')

        # Initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if ratings_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])


            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))

        # Validation
        print('Start validation ...')
        rmse, perc_rated, actual_v_pred, preds, acts = self.validation_comparison(validation_df, user_mat=user_mat, movie_mat=movie_mat)
        print('rmse: ', rmse)
        print('perc_rated: ', perc_rated)
        print('actual_v_pred: ', actual_v_pred)

        self.plot_validation_results(rmse, perc_rated, actual_v_pred, preds, acts)

        print(' ')
        print('Saving user-by-item matrix as pickle ...')
        with open('ratings_mat.pkl','wb') as f:
            pickle.dump(ratings_mat, f)
        print('...done')
        print('------------------------')
        print(' ')

        print(' ')
        print('Saving user_mat as pickle ...')
        with open('user_mat.pkl','wb') as f:
            pickle.dump(user_mat, f)
        print('...done')
        print('------------------------')
        print(' ')

        print(' ')
        print('Saving movie_mat as pickle ...')
        with open('movie_mat.pkl','wb') as f:
            pickle.dump(movie_mat, f)
        print('...done')
        print('------------------------')
        print(' ')

        return user_mat, movie_mat, ratings_mat

    def predict_rating(self, user_matrix, movie_matrix, user_id, movie_id, load_mat=False):
        ''' makes predictions of a rating for a user on a movie-user combo

        INPUTS:
        ------------
            user_matrix - user by latent factor matrix
            movie_matrix - latent factor by movie matrix
            user_id - the user_id from the reviews df
            movie_id - the movie_id according the movies df

        OUTPUTS:
        ------------
            pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        if load_mat==True:
            ratings_mat, user_mat, movie_mat, ratings_mat = self.load_matrices()

        # Create series of users and movies in the right order
        user_ids_series = np.array(ratings_mat.index)
        movie_ids_series = np.array(ratings_mat.columns)

        # User row and Movie Column
        user_row = np.where(user_ids_series == user_id)[0][0]
        movie_col = np.where(movie_ids_series == movie_id)[0][0]

        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])

        return pred

    def validation_comparison(self, val_df, user_mat, movie_mat):
        '''
        INPUTS:
        ------------
            val_df - the validation dataset created in create_train_test
            user_mat - U matrix in FunkSVD
            movie_mat - V matrix in FunkSVD

        OUTPUTS:
        ------------
            rmse - RMSE of how far off each value is from it's predicted value
            perc_rated - percent of predictions out of all possible that could be rated
            actual_v_pred - a 10 x 10 grid with counts for actual vs predicted values
        '''

        val_users = np.array(val_df['user_id'])
        val_movies = np.array(val_df['movie_id'])
        val_ratings = np.array(val_df['rating'])

        sse = 0
        num_rated = 0
        preds, acts = [], []
        actual_v_pred = np.zeros((10,10))
        print(len(len(val_users)))
        for idx in range(len(val_users)):
            print(idx)
            try:
                print('idx not null ', idx)
                pred = self.predict_rating(user_mat, movie_mat, val_users[idx], val_movies[idx])
                sse += (val_ratings[idx] - pred)**2
                num_rated+=1
                preds.append(pred)
                acts.append(val_ratings[idx])
                actual_v_pred[11-int(val_ratings[idx]-1), int(round(pred)-1)]+=1

            except:
                continue

        rmse = np.sqrt(sse/num_rated)
        perc_rated = num_rated/len(val_users)
        return rmse, perc_rated, actual_v_pred, preds, acts

    def plot_validation_results(self, rmse, perc_rated, actual_v_pred, preds, acts):
        # How well did we do?
        print(rmse, perc_rated)
        sns.heatmap(actual_v_pred);
        plt.xticks(np.arange(10), np.arange(1,11));
        plt.yticks(np.arange(10), np.arange(1,11));
        plt.xlabel("Predicted Values");
        plt.ylabel("Actual Values");
        plt.title("Actual vs. Predicted Values");

    def load_matrices(self, ratings_mat_path='ratings_mat.pkl', user_mat_path='user_mat.pkl', movie_mat_path='movie_mat.pkl'):

        with open(ratings_mat_path,'rb') as f:
            ratings_mat = pickle.load(f)
        print('Shape of user_mat')
        print(ratings_mat.shape)
        print('------------------------')
        print(' ')

        with open(user_mat_path,'rb') as f:
            user_mat = pickle.load(f)
        print('Shape of user_mat')
        print(user_mat.shape)
        print('------------------------')
        print(' ')

        with open(movie_mat_path,'rb') as f:
            movie_mat = pickle.load(f)
        print('Shape of user_mat')
        print(movie_mat.shape)
        print('------------------------')
        print(' ')

        return ratings_mat, user_mat, movie_mat


    def find_similar_movies(self, movie_id):
        '''
        INPUTS:
        ------------
            movie_id - a movie_id

        OUTPUTS:
        ------------
            similar_movies - an array of the most similar movies by title
        '''

        # find the row of each movie id
        movie_idx = np.where(movies['movie_id'] == movie_id)[0][0]

        # find the most similar movie indices - to start I said they need to be the same for all content
        similar_idxs = np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]

        # pull the movie titles based on the indices
        similar_movies = np.array(movies.iloc[similar_idxs, ]['movie'])

        return similar_movies

    def get_movie_names(self, movie_ids):
        '''
        INPUTS:
        ------------
            movie_ids - a list of movie_ids

        OUTPUT:
        ------------
            movies - a list of movie names associated with the movie_ids
        '''

        movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])

        return movie_lst



    def create_ranked_df(self, movies, reviews):
        '''
        INPUTS:
        ------------
            movies - the movies dataframe
            reviews - the reviews dataframe

        OUTPUT:
        ------------
            ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews,
                        then time, and must have more than 4 ratings
        '''

        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

        return ranked_movies


    def popular_recommendations(self, user_id, n_top, ranked_movies):
        '''
        INPUT:
        ------------
            user_id - the user_id (str) of the individual you are making recommendations for
            n_top - an integer of the number recommendations you want back
            ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

        OUTPUTS:
        ------------
            top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''

        top_movies = list(ranked_movies['movie'][:n_top])

        return top_movies



    def start_prediction(self):
        user_mat, movie_mat = self.load_matrices()


    def make_recs(self, _id, train_data, train_df, movies, user_mat, _id_type='movie', rec_num=5):
        '''
        INPUTS:
        ------------
            _id - either a user or movie id (int)
            _id_type - "movie" or "user" (str)
            train_data - dataframe of data as user-movie matrix
            train_df - dataframe of training data reviews
            movies - movies df
            user_mat - the U matrix of matrix factorization
            movie_mat - the V matrix of matrix factorization
            rec_num - number of recommendations to return (int)

        OUTPUTS:
        ------------
            recs - (array) a list or numpy array of recommended movies like the
                    given movie, or recs for a user_id given
        '''

        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        val_users = train_data_df.index
        rec_ids = create_ranked_df(movies, train_df)

        if _id_type == 'user':
            if _id in train_data.index:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(val_users == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(user_mat[idx,:],movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = train_data_df.columns[indices]
                rec_names = get_movie_names(rec_ids)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = popular_recommendations(_id, rec_num, ranked_movies)

        # Find similar movies if it is a movie that is passed
        else:
            rec_ids = find_similar_movies(_id)
            rec_names = get_movie_names(rec_ids)

        return rec_ids, rec_names



if __name__ == '__main__':
    # test different parts to make sure it works
    pass
