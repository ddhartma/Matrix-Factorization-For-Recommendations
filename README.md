[image1]: assets/val_practice.png "image1"
[image2]: assets/train_test_split.png "image2"
[image10]: assets/svd.png "image10"
[image3]: assets/user_item_matrix.png "image3"
[image4]: assets/latent_factors.png "image4"
[image5]: assets/uet.png "image5"
[image6]: assets/u_mat.png "image6"
[image7]: assets/v_trans_mat.png "image7"
[image8]: assets/sigma_mat.png "image8"
[image9]: assets/u_sigma_v_together.png "image9"
[image11]: assets/svd_take_away.png "image11"
[image12]: assets/loc_based_recom.png "image12"
[image13]: assets/deep_learn_recom.png "image13"
[image14]: assets/funk_svd_1.png "image14"
[image15]: assets/funk_svd_2.png "image15"
[image16]: assets/funk_svd_3.png "image16"

# Matrix Factorization for Recommendation 
In this lesson, you will learn about three main topics:

- We will look at validating recommendations (at a high level).
- We will look at matrix factorization as a method to use machine learning for recommendations.
- We will look at combining recommendation techniques to make predictions to existing and new users and for existing and new items.



## Outline
- [Validating Recommendations](#Recommendation_Engines)
- [Singular Value Decomposition (SVD) ](#SVD)
    - [Traditional SVD](#trad_SVD)
    - [The SVD concept](#SVD_concept)
    - [SVD in Code](#SVD_in_code)
    - [SVD - Takeaway message](#SVD_Takeaway_message)

- [Funk SVD](#Funk_SVD)
    - [Funk SVD principle](#Funk_SVD_principle)
    - [Funk SVD in Code](#Funk_SVD_in_Code)
    - How are 
  
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Validating Recommendations <a name="Recommendation_Engines"></a>

- How do we know, if our users like the recommendations?
1. ***Online Testing***:
    - Before/After Test
    - We could look at some metrics of interest prior to implementing the recommendation
        - User engagement
        - Downlodas
        - Revenue
    - Look at the metric again after implementing the recommendation engine 
    - Is there any impact?

    - Similar: A/B Test

    ![image1]

2. ***Offline Testing***: How could we proof the performance of the Recommendation Engine before using it on real users?
    - ***Split*** data into a training and testing partition
    - ***Fit*** the recommender on the training set
    - ***Evaluate*** the performance on the testing set
    - ***Sort***: In case of collected ratings ***over time*** 
        - Newest data --> for testing set
        - Older data --> for training set
        - This avoids using future data for making predictions on past data
    - Idea: 
        - ***Predict*** ratings (user-item-combination) for every movie 
        - ***Compare*** the ratings of a certain user to our predictions 
        - ***Understand***: If we do this for every rating in the test set we can understand how well our recommendation engine is working

![image2]

3. ***User Groups***:
    - Having user groups which give feedback on items you would recommend for them.
    - Obtaining good user groups that are representative of your customers can be a challenge on its own.
    - This is especially true when you have a lot of products and a very large consumer base. 

# Singular Value Decomposition (SVD) <a name="SVD"></a>
Let's start with traditional SVDs

![image10]

## Traditional SVD: <a name="trad_SVD"></a>
- ***Latent Factors***:
    - A traditional approach for matrix factorization
    - When performing SVD, we create a matrix of users by items (or customers by movies in our specific example), with user ratings for each item scattered throughout the matrix. An example is shown in the image below.
    - This matrix doesn't have any specific information about the users or items. Rather, it just holds the ratings that each user gave to each item. Using SVD on this matrix, we can find latent features related to the movies and customers. This is amazing because the dataset doesn't contain any information about the customers or movies!

    ![image3]

    - A Latent Factor is not observed in the data, but we infer it based on the ratings users give to items
    - Finding how items (movies) and user relate to Latent Factors is central for making predictions with SVD

    ![image4]

## The SVD concept: <a name="SVD_concept"></a>

- only works when there are no missing values
- r represents the rating of some user for some rating
- first index (1..n) -- for user
- second index (1...m) -- for item

    ![image5]

- ***U matrix***: 
    - Info about how users are related to particular latent factors
    - numbers indicates how each user feels about each latent factor
    - n rows -- users
    - k columns -- latent factors

    ![image6]

- ***V-transpose matrix***: 
    - Info about how latent factors are related to items (movies)
    - the higher the value the stronger the relationship 
    - e.g. A.I. and WALL E are strongly related to the robot latent feature
    - k rows -- latent factors
    - m columns -- movies

    ![image7]

- ***Sigma matrix***:
    - k x k diagonal matrix
    - only diagonal elements are not zero
    - same number of rows and columns as number of latent factors 
    - values in the diagonal are always positive and sorted from largest to smallest 
    - the diagonal indicated how many latent factors we want to keep 
    - first weight is associated with the first latent factor
    - if the the weights are larger, this is an indication that the correponding latent factor is more important to reproduce the ratings of the original user item matrix 
    - here: the fact that the movie is related to dogs is more important in preticting ratings than using prefferences on robots or sadness

    ![image8]
    
- ***Setting all together***
    - By multiplying these matrices together we are reconstructing a movie rating for each user-movie combination based on how the users feel about the latent factors.
   - By finding values for U, Sigma and V-transpose, we can find a prediction for every user-movie combination

    ![image9]

## SVD in Code <a name="SVD_in_code"></a>
-  Open notebook ```./notebooks/1_Intro_to_SVD.ipynb```

    ```
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import svd_tests as t
    %matplotlib inline

    # Read in the datasets
    movies = pd.read_csv('data/movies_clean.csv')
    reviews = pd.read_csv('data/reviews_clean.csv')

    del movies['Unnamed: 0']
    del reviews['Unnamed: 0']

    # Create user-by-item matrix
    user_items = reviews[['user_id', 'movie_id', 'rating']]
    user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

    user_movie_subset = user_by_movie[[73486, 75314,  68646, 99685]].dropna(axis=0)
    print(user_movie_subset)
    ```
    Create SVD via numpy.linalg.svd 
    ```
    u, s, vt = np.linalg.svd(user_movie_subset)
    s.shape, u.shape, vt.shape
    ```
    Change dimensions of u, s and vt 
    ```
    # Change the dimensions of u, s, and vt as necessary to use four latent features
    # update the shape of u and store in u_new
    u_new = u[:, :len(s)]

    # update the shape of s and store in s_new
    s_new = np.zeros((len(s), len(s)))
    s_new[:len(s), :len(s)] = np.diag(s) 

    # Because we are using 4 latent features and there are only 4 movies, 
    # vt and vt_new are the same
    vt_new = vt
    ```
    Check if the matrix product of u, s and vt is the gives back the original user-movie matrix
    ```
    np.allclose(np.dot(np.dot(u_new, s_new), vt_new), user_movie_subset)
    ```
    Check how much of the variability can be explained by principal compoments of s. The total amount of variability to be explained is the sum of the squared diagonal elements. The amount of variability explained by the first componenet is the square of the first value in the diagonal. The amount of variability explained by the second componenet is the square of the second value in the diagonal. 
    ```
    total_var = np.sum(s**2)
    var_exp_comp1_and_comp2 = s[0]**2 + s[1]**2
    perc_exp = round(var_exp_comp1_and_comp2/total_var*100, 2)
    print("The total variance in the original matrix is {}.".format(total_var))
    print("Ther percentage of variability captured by the first two components is {}%.".format(perc_exp))
    ```
    As 98.55% of variability can be explained by the first two components reduce s to a 2x2 matrix.
    ```
    # Change the dimensions of u, s, and vt as necessary to use four latent features
    # update the shape of u and store in u_new
    k = 2
    u_2 = u[:, :k]

    # update the shape of s and store in s_new
    s_2 = np.zeros((k, k))
    s_2[:k, :k] = np.diag(s[:k]) 

    # Because we are using 2 latent features, we need to update vt this time
    vt_2 = vt[:k, :]
    ```
    Calculate the MSE
    ```
    # Compute the dot product
    pred_ratings = np.dot(np.dot(u_2, s_2), vt_2)

    # Compute the squared error for each predicted vs. actual rating
    sum_square_errs = np.sum(np.sum((user_movie_subset - pred_ratings)**2))
    ```

## SVD - Takeaway message <a name="SVD_Takeaway_message"></a>

- Three main takeaways from the previous notebook:

    - The latent factors retrieved from SVD aren't actually labeled.
    - We can get an idea of how many latent factors we might want to keep by using the Sigma matrix.
    - SVD in NumPy will not work when our matrix has missing values. This makes this technique less than useful for our current user-movie matrix.

    ![image11]


# Funk SVD <a name="Funk_SVD"></a>
- A slight modification of traditional SVD work incredibly well during the [Netflix competition](https://en.wikipedia.org/wiki/Netflix_Prize)
- It is one of the most popular recommendation approaches in use today
- It works for situtations with ***missing values***

With Funk SVD one can compute missing values from only the given values
- Take two matrices U and Vt
- Fill them randomly with values
- Take from U ratings that already exist 

## Funk SVD principle <a name="Funk_SVD_principle"></a>

Use the following rule in updating these random values 
- Take the ***dot product*** between 
    - the ***row asssociated with the user*** and
    - the ***column associated with the movie***
    - to get a ***prediction for the rating*** (here 0.4)
- Calculate the ***error*** as the difference between the actual and the predicted rating. 
- The ***goal*** is to ***minimize this error*** across all ***known*** values by changing the weights in each matrix.
- Technique to be used: ***Gradient Descent***
    - Calculate the derivatives of the error with respect to each value identified by u and v, respectively. 
    - Use the chain rule to find the direction to move on to minimize the error in each matrix. 
    - y - uv is the difference between the actual and predicted value 
    - use the update rule to calculate new updated values for the u and v matrix
        - move in the opposite direction of the gardient to minimize the error for the next iteration step.
        - use a learning rate alpha to take a small step in this direction 

    ![image14]

    ![image15]
    
    ![image16]

## Funk SVD in Code <a name="Funk_SVD_in_Code"></a>

- Open notebook ```./notebooks/2_Implementing_FunkSVD.ipynb```
- Open notebook ```./notebooks/3_How_well_we_are_doing.ipynb```

    ```
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import sparse
    import svd_tests as t
    %matplotlib inline

    # Read in the datasets
    movies = pd.read_csv('data/movies_clean.csv')
    reviews = pd.read_csv('data/reviews_clean.csv')

    del movies['Unnamed: 0']
    del reviews['Unnamed: 0']
    ```

    ```
    def create_train_test(reviews, order_by, training_size, testing_size):
        '''    
        INPUTS:
        ------------
            reviews - (pandas df) dataframe to split into train and test
            order_by - (string) column name to sort by
            training_size - (int) number of rows in training set
            testing_size - (int) number of columns in the test set
        
        OUTPUTS:
        ------------
            training_df -  (pandas df) dataframe of the training set
            validation_df - (pandas df) dataframe of the test set
        '''
        reviews_new = reviews.sort_values(order_by)
        training_df = reviews_new.head(training_size)
        validation_df = reviews_new.iloc[training_size:training_size+testing_size]
        
        return training_df, validation_df
    ```

    ```
    def FunkSVD(ratings_mat, latent_features=12, learning_rate=0.0001, iters=100):
        ''' This function performs matrix factorization using a basic form of FunkSVD with no regularization
        
        INPUTS:
        ------------
            ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
            latent_features - (int) the number of latent features used
            learning_rate - (float) the learning rate 
            iters - (int) the number of iterations
        
        OUTPUTS:
        ------------
            user_mat - (numpy array) a user by latent feature matrix
            movie_mat - (numpy array) a latent feature by movie matrix
        '''
        
        # Set up useful values to be used through the rest of the function
        n_users = ratings_mat.shape[0]
        n_movies = ratings_mat.shape[1]
        num_ratings = np.count_nonzero(~np.isnan(ratings_mat))
        
        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(n_users, latent_features)
        movie_mat = np.random.rand(latent_features, n_movies)
        
        # initialize sse at 0 for first iteration
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
            for i in range(n_users):
                for j in range(n_movies):
                    
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
            print("%d \t\t %f" % (iteration+1, sse_accum / num_ratings))
            
        return user_mat, movie_mat 
    
    # Create user-by-item matrix
    train_user_item = train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
    train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
    train_data_np = np.array(train_data_df)

    # Fit FunkSVD with the specified hyper parameters to the training data
    user_mat, movie_mat = FunkSVD(train_data_np, latent_features=15, learning_rate=0.005, iters=250)    
    ```
    ```
    def predict_rating(user_matrix, movie_matrix, user_id, movie_id):
        '''
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
        
        # Use the training data to create a series of users and movies that matches the ordering in training data
        user_ids_series = np.array(train_data_df.index)
        movie_ids_series = np.array(train_data_df.columns)
        
        # User row and Movie Column
        user_row = np.where(user_ids_series == user_id)[0][0]
        movie_col = np.where(movie_ids_series == movie_id)[0][0]
        
        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])
        
        return pred

    # Test your function with the first user-movie in the user-movie matrix (notice this is a nan)
    pred_val = predict_rating(user_mat, movie_mat, 8, 2844)
    pred_val
    ```
    ```
    def print_prediction_summary(user_id, movie_id, prediction):
        '''
        INPUTS:
        ------------
            user_id - the user_id from the reviews df
            movie_id - the movie_id according the movies df
            prediction - the predicted rating for user_id-movie_id
        OUTPUTS:
        ------------
            None
        '''
        
        movie_name = str(movies[movies['movie_id'] == movie_id]['movie']) [5:]
        movie_name = movie_name.replace('\nName: movie, dtype: object', '')
        print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(prediction, 2), str(movie_name)))

    # Test your function the the results of the previous function
    print_prediction_summary(8, 2844, pred_val)
    ```
    ```
    def validation_comparison(val_df, num_preds):
        '''
        INPUTS:
        ------------
            val_df - the validation dataset created in the third cell above
            num_preds - (int) the number of rows (going in order) you would like to make predictions for

        OUTPUTS:
        ------------
            Nothing returned - print a statement about the prediciton made for each row of val_df from row 0 to num_preds±
        '''
        
        val_users = np.array(val_df['user_id'])
        val_movies = np.array(val_df['movie_id'])
        val_ratings = np.array(val_df['rating'])
        
        
        for idx in range(num_preds):
            pred = predict_rating(user_mat, movie_mat, val_users[idx], val_movies[idx])
            print("The actual rating for user {} on movie {} is {}.\n While the predicted rating is {}.".format(val_users[idx], val_movies[idx], val_ratings[idx], round(pred))) 

            
    # Perform the predicted vs. actual for the first 6 rows.  How does it look?
    validation_comparison(val_df, 6) 
    ```
## The Cold Start Problem

## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a Command Line Interface (CLI) under Windows you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Recommendation-Engines.git
```

- Change Directory
```
$ cd Recommendation-Engines
```

- Create a new Python environment, e.g. rec_eng. Inside Git Bash (Terminal) write:
```
$ conda create --name rec_eng
```

- Activate the installed environment via
```
$ conda activate rec_eng
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
Recommendation Engines
* [Essentials of recommendation engines: content-based and collaborative filtering](https://towardsdatascience.com/essentials-of-recommendation-engines-content-based-and-collaborative-filtering-31521c964922)
* [AirBnB uses Embeddings for Recommendations](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
* [Location-Based Recommendation Systems](https://link.springer.com/referenceworkentry/10.1007%2F978-3-319-17885-1_1580)

    ![image12]
* [Deep learning for recommender systems](https://ebaytech.berlin/deep-learning-for-recommender-systems-48c786a20e1a)

    ![image13]

* [Dimensionality Reduction](http://infolab.stanford.edu/~ullman/mmds/ch11.pdf±±±±±)
* [PCA](https://de.wikipedia.org/wiki/Hauptkomponentenanalyse)
* (https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
