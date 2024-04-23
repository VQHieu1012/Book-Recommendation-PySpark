from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Window
from pyspark.sql.functions import rank, col, lit
import streamlit as st
import pandas as pd
import numpy as np
import pickle


### Kmeans
cluster_0 = pd.read_csv('./data/cluster_0.csv')
cluster_1 = pd.read_csv('./data/cluster_1.csv')
top_ratings_cluster_0 = pd.read_csv('./data/top_ratings_cluster_0.csv')
top_ratings_cluster_1 = pd.read_csv('./data/top_ratings_cluster_1.csv')
ratings_kmeans = pd.read_csv('./data/rating_indexed.csv')

def recommend_kmeans(user_id):
    """
    user_id (int): user's id
     --> cluster_0 / cluster_1: [id, scaledFeature, prediction, distance_to_centroid, ISBN, rating]
    
    Return: list of ISBNs
    """

    user_id = int(user_id)
    is_present_0 = cluster_0['id'].isin([user_id]).any()
    # Get all book user_id has rated
    books_rated_by_user = ratings_kmeans[ratings_kmeans['user_id'] == user_id]
    if is_present_0:
        cluster = 0
        recommend = top_ratings_cluster_0[~top_ratings_cluster_0["ISBN"].isin(books_rated_by_user['ISBN'].values)]

    is_present_1 = cluster_1['id'].isin([user_id]).any()
    if is_present_1:
        cluster = 1
        recommend = top_ratings_cluster_1[~top_ratings_cluster_1["ISBN"].isin(books_rated_by_user['ISBN'].values)]    

    return cluster, recommend['ISBN'].values[:10]

### END Kmeans ###

### Spark Setup and transform 
spark = SparkSession.builder.appName("Recommendation").getOrCreate()
model_path = "./model/model_2"
model = ALSModel.load(model_path)
ratings_ALS = spark.read.csv('./data/rating_indexed.csv', header=True, inferSchema=True)
pred = model.transform(ratings_ALS)

### Prepare data
books = pd.read_csv('./data/cleaned_books.csv')


### Function in ALS ###

def recommend_als(user_id):
    """
    user_id (int): user's id
    ---> predictions: [item_id: double, user_id: int, prediction: float]
    ---> ratings_ALS: [user_id: int, ISBN: string, rating: int, item_id: double]
    
    Return: list of ISBN
    """
    selected_user_id = int(user_id)
    user_unrated_items = ratings_ALS.filter(col("user_id") != selected_user_id).select("item_id").distinct()

    # Dự đoán xếp hạng cho các item chưa đánh giá của người dùng được chọn
    predictions = model.transform(user_unrated_items.withColumn("user_id", lit(selected_user_id)))

    # Sắp xếp kết quả theo xếp hạng dự đoán giảm dần và chọn top N item
    top_n_recommendations = predictions.orderBy("prediction", ascending=False).limit(10)

    # Hiển thị kết quả gợi ý
    re = top_n_recommendations.select('item_id').collect()
    item_ids = [i[0] for i in re]
    ISBNs = ratings_ALS.filter(ratings_ALS['item_id'].isin(item_ids))
    recommend = [i[0] for i in ISBNs.select('ISBN').distinct().collect()]
    return recommend

# Get support funtion
def get_info_isbn(isbn):
    data = books[books['ISBN'].isin([isbn])]
    title, author, publisher, img = data['title'].values[0], data['author'].values[0], data['publisher'].values[0], data['Image-URL-S'].values[0]
    return (title, author, publisher, img)

# item-item
with open('./model/knn_model.pkl', 'rb') as file:
    nn_model = pickle.load(file)
df = pd.read_parquet('./data/item_factor.parquet/')

indexed_ratings = pd.read_csv('./data/rating_indexed.csv')
def recommend_item_item(book_id):
    book_id = int(book_id)
    feature = df[df['id']==book_id]['features'].tolist()
    distance, suggestion = nn_model.kneighbors(feature, n_neighbors=11)
    ISBN = indexed_ratings[indexed_ratings['item_id'].isin(suggestion[0][1:])]['ISBN'].unique().tolist()
    return ISBN

# Streamlit
st.header("Book recommender system")

#indexed_ratings = pd.read_csv('./data/rating_indexed.csv')

#result = pd.merge(indexed_ratings, books, on='ISBN')

#book_names = result['title'].unique()
#user_ids = result['user_id'].unique()
user_factor = pd.read_parquet('./data/user_factor.parquet/')
user_ids = user_factor['id'].unique()

item_factor = pd.read_parquet('./data/item_factor.parquet/')
item_ids = item_factor['id'].unique()


def info(book_id):
    book_id = int(book_id)
    ISBN = indexed_ratings[indexed_ratings['item_id']==book_id]['ISBN'].values[0]

    #ISBN = '0340590262'
    try:
        title, author, publisher, img = get_info_isbn(ISBN)
        st.text(title)
        st.text(author)
        st.text(publisher)
        st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(ISBN))
    except:
        st.text("No infor for "+ ISBN)
        st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(ISBN))

selected_books = st.selectbox(
    "Type or select a book",
    np.append(item_ids, "None"),
)

btn1 = st.button('Similar book: ')
if btn1:
    info(selected_books)

    result_list = recommend_item_item(selected_books)
    
    col_num = 5
    col0 = st.columns(col_num)
    for i in range(0, 5):
            with col0[i]:
                try:
                    title, author, publisher, img = get_info_isbn(result_list[i])
                    st.text(title)
                    st.text(author)
                    st.text(publisher)
                    #st.image(img)
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i]))
                except:
                    st.text("No infor for "+ result_list[i])
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i]))
    col1 = st.columns(col_num)
    for i in range(0, 5):
        with col1[i]:
            try:
                title, author, publisher, img = get_info_isbn(result_list[i+5])
                st.text(title)
                st.text(author)
                st.text(publisher)
                #st.image(img)
                st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))
            except:
                st.text("No infor for " + result_list[i+5])
                st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))

selected_users = st.selectbox(
    "Type or select an user",
    np.append(user_ids, "None")
)

btn = st.button('------')


btn2 = st.button('Show top 5 items for user id '+ selected_users+" using ALS")
if btn2 or btn:
    if selected_users != "None":
        
        result_list = recommend_als(selected_users)
    
        col_num = 5
        col0 = st.columns(col_num)
        for i in range(0, 5):
            with col0[i]:
                title, author, publisher, img = get_info_isbn(result_list[i])
                st.text(title)
                st.text(author)
                st.text(publisher)
                #st.image(img)
                st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i]))

        col1 = st.columns(col_num)
        for i in range(0, 5):
            with col1[i]:
                title, author, publisher, img = get_info_isbn(result_list[i+5])
                st.text(title)
                st.text(author)
                st.text(publisher)
                #st.image(img)
                st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))
    else:
        st.write("Select user id.")

    
btn3 = st.button('Recommend books for: '+ selected_users)
if btn3 or btn:
    if selected_users != "None":
        
        cluster, result_list = recommend_kmeans(selected_users)
        st.write(selected_users + " in cluster " + str(cluster))
        col_num = 5
        col0 = st.columns(col_num)
        for i in range(0, 5):
            with col0[i]:
                try:
                    title, author, publisher, img = get_info_isbn(result_list[i])
                    st.text(title)
                    st.text(author)
                    st.text(publisher)
                    #st.image(img)
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i]))
                except:
                    st.text("No infor for "+ result_list[i])
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))
        col1 = st.columns(col_num)
        for i in range(0, 5):
            with col1[i]:
                try:
                    title, author, publisher, img = get_info_isbn(result_list[i+5])
                    st.text(title)
                    st.text(author)
                    st.text(publisher)
                    #st.image(img)
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))
                except:
                    st.text("No infor for " + result_list[i+5])
                    st.image('https://covers.openlibrary.org/b/isbn/{}-M.jpg'.format(result_list[i+5]))
    else:
        st.write("Select user id.")

#st.image('https://pictures.abebooks.com/isbn/9782908730302-us.jpg')