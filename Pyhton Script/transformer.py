import numpy as np
import pandas as pd
import psycopg2
import subprocess
import sys
from sklearn.metrics.pairwise import linear_kernel
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_user_info(user_id):

    user_data = int(user_id)
    return user_data

if __name__ == "__main__":

    # Kullanım: python script.py user_id
    if len(sys.argv) != 2:

        sys.exit(1)

    user_id_param = sys.argv[1]

    user_info = get_user_info(user_id_param)

    if user_info:
        print("Kullanıcı Bilgileri:",user_info)




"""GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ"""

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    'numpy',
    'pandas',
    'psycopg2',
    'scikit-learn',
    'torch',
    'sentence-transformers',
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} bulunamadı. Yükleniyor...")
        install_package(package)

print("Gerekli kütüphaneler yüklendi.")

"""SQL BAĞLANTISI"""

DATABASE_URI = 'postgresql+psycopg2://jmmbsbkmeuckih:d96b43e0c9cc786775c616abd0f3f84f5ba3720ff38881f01d2d617e4889b913@ec2-52-31-161-46.eu-west-1.compute.amazonaws.com/d6cfnvoqmk3rqr'

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()



"""Recommendations tablosundaki veriyi silme fonksiyonu """

def clear_recommendations(session):
    try:
        session.execute(text("DELETE FROM recommendations"))
        session.commit()
        print("Tüm öneriler başarıyla silindi.")
    except Exception as e:
        print(f"Hata: {e}")


'''DATABASE VERİLERİNİ ÇEKME'''

# Film verilerini çekme

engine = session.bind
connection = engine.raw_connection()
query_movie='SELECT "movie_id", "title", "genre" FROM movie'
movie_data = pd.read_sql_query(query_movie, connection)
header_movie = ['movie_id', 'title', 'genre']
movie_data.columns = header_movie
connection.close()


# Kullanıcının tıkladığı son filmi bulma
engine = session.bind
connection = engine.raw_connection()
clicked_last = pd.read_sql_query('SELECT "userId", "movieId" FROM movie_clicks WHERE "userId" = %s', connection, params=[int(user_info)])


# Puan verilerini çekme
engine = session.bind
connection = engine.raw_connection()
query_rate='SELECT "userId", "movie_id", "rating", "timestamp" FROM rating'
rating_data = pd.read_sql_query(query_rate, connection)
header_rating = ['user_id', 'item_id', 'rating', 'timestamp']
rating_data.columns = header_rating


# Kullanıcının puan verdiği filmleri bulma
engine = session.bind
connection = engine.raw_connection()
rr_query = pd.read_sql_query('SELECT "userId", "movie_id", "rating", "timestamp" FROM rating WHERE "userId" = %s',connection,params= [ int(user_info)])


"""""Recom tablosu verilerini sil """

clear_recommendations(session)

"""KULLANICININ OY KULLANIP KULLANMADIĞINI BUL"""
def is_user_in_ratings(session, user_id):
    try:
        result = session.execute(text('SELECT * FROM rating WHERE "userId" = :user_id LIMIT 1'), {'user_id': user_id}).fetchone()
        return result is not None
    except Exception as e:
        print(f"Hata: {e}")
        return False

# Kontrolü yap
user_in_ratings = is_user_in_ratings(session, user_info)


"""KULLANICININ FİLMLERE TIKLAYIP TIKLAMADIĞINI ÖĞRENME"""
def is_user_in_clicks(session, user_id):
    try:
        result = session.execute(text('SELECT * FROM movie_clicks WHERE "userId" = :user_id LIMIT 1'), {'user_id': user_id}).fetchone()
        return result is not None
    except Exception as e:
        print(f"Hata: {e}")
        return False

# Kontrolü yap
user_in_clicks = is_user_in_clicks(session, user_info)

"""" EŞSİZ KULLANICI VE FİLM SAYISI BELİRLEME """

n_users = rating_data.user_id.unique().shape[0]
n_items = rating_data.item_id.unique().shape[0]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""TRANSFORMER YAPILI İÇERİK TABANLI ÖNERİ """""""

# Film verilerini çekme
engine = session.bind
connection = engine.raw_connection()
query_transform="SELECT * FROM movie"
transformer_data = pd.read_sql_query(query_transform, connection)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(transformer_data['description'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


results = {}
for idx, row in transformer_data.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], transformer_data['movie_id'][i]) for i in similar_indices]
    results[row['movie_id']] = similar_items[1:]

def item(id):
    return transformer_data.loc[transformer_data['movie_id'] == id]['title'].tolist()[0].split(' - ')[0]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

"""""""GÖMME VEKTÖRLERİ OLUŞTURMA"""""""

#descriptions = transformer_data['description'].tolist()
#print(descriptions)
#des_embeddings = []
#for i,des in enumerate(descriptions):
    #des_embeddings.append(model.encode(des))

#embedding_array = np.array(des_embeddings)

#np.save('embedding_array.npy', embedding_array)


#ÖNCEDEN VAR İSE DİREKT YÜKLEYEBİLİRİZ

des_embeddings = np.load('C:/Users/bckal/Desktop/Project 491/pythonProject/embedding_array.npy')

def recommend(query):
    #yerleştirmelerle kosinus benzerliği hesaplama
    #4 adet film önerir
    query_embedd = model.encode(query)
    cosine_scores = util.pytorch_cos_sim(query_embedd, des_embeddings)
    top_matches = torch.argsort(cosine_scores, dim=-1, descending=True).tolist()[0][1:5] #6
    return top_matches


'''COLLOBRATIVE FILTERİNG İLE ÖNERİ AL'''
def collaborative_recommendations(user_id, min_count):

    engine = session.bind
    connection = engine.raw_connection()
    movie_query = 'SELECT "movie_id", "date_added", "title" FROM movie;'
    movie_data = pd.read_sql_query(movie_query, connection)



    movie_data.set_index("movie_id", inplace=True)

    f = ["count", "mean"]

    engine = session.bind
    connection = engine.raw_connection()
    coll_rating_query = 'SELECT "rating", "userId", "movie_id" FROM rating;'
    coll_rating=pd.read_sql_query(coll_rating_query, connection)



    df_movie_summary = coll_rating.groupby("movie_id")["rating"].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)

    df_p = pd.pivot_table(coll_rating, values="rating", index="userId", columns="movie_id")
    df_p.shape


    # Select ratings given by the user
    target_user_ratings = df_p.loc[user_id].dropna()

    last_id_clicked = target_user_ratings.index[-1]


     #If user rates more than 1 movie

    if len(target_user_ratings)>1:


    # Calculate correlation with other movies
     similar_to_target_user = df_p.corrwith(target_user_ratings)


     # Create a DataFrame with correlation values
     corr_target_user = pd.DataFrame(similar_to_target_user, columns=["PearsonR"])
     corr_target_user["movie_id"] = similar_to_target_user.index
     corr_target_user.dropna(inplace=True)

     # Sort the DataFrame by correlation values in descending order
     corr_target_user = corr_target_user.sort_values("PearsonR", ascending=False)

     # Map indices to integers
     corr_target_user.index = corr_target_user.index.map(int)

     # Join with movie and summary information
     corr_target_user = corr_target_user.join(movie_data).join(df_movie_summary)[
         ["PearsonR","movie_id", "title", "count", "mean"]
     ]

     if corr_target_user.empty:
         return last_id_clicked


     return corr_target_user

    elif len(target_user_ratings)==1:


        return last_id_clicked
    else:

        return last_id_clicked



'''ÖNERİ OLUŞTURMA VE YOLLAMA'''
def pearson_recommendations(user_id):

    engine = session.bind
    connection = engine.raw_connection()


    user_in_clicks = is_user_in_clicks(session, user_info)


    user_in_ratings = is_user_in_ratings(session, user_info)

    print(user_in_ratings)


    if user_in_ratings:

      cf_rec= collaborative_recommendations(user_id,0)


      if isinstance(cf_rec, np.int64) and user_in_clicks is False:

        clicked_last = cf_rec
        movie_id_to_use = clicked_last

        id = movie_id_to_use
        query_show_des = transformer_data.loc[transformer_data['movie_id'] == id]['description'].to_list()[0]
        if query_show_des:
            # query_show_des boş değilse devam et
            query_show_des = query_show_des[0]
            recommendded_results = recommend(query_show_des)

            title_transformer_rec = []
            for index in recommendded_results:
                if 0 <= index < len(transformer_data):
                    movie_id_transformer_rec = transformer_data.iloc[index]['movie_id']
                    title_transformer_rec.append(transformer_data.iloc[index]['title'])
                    show_id = transformer_data.iloc[index]['movie_id']
                    title = transformer_data.iloc[index]['title']
                    print("ID:", show_id, "Title:", title)
                    session.execute(text('INSERT INTO recommendations (show_id, title) VALUES (:show_id, :title)'),
                                    {'show_id': int(show_id), 'title': str(title)})
                    session.commit()
                else:
                    print(f"Invalid index: {index}")
                    connection.close()

        return


      elif isinstance(cf_rec, int):
          collob_rec=cf_rec
          return collob_rec

      elif isinstance(cf_rec,pd.DataFrame):
          collob_rec=cf_rec[:4]
          return collob_rec

      else:
          if isinstance(cf_rec, list):
              collob_rec = cf_rec[:3]  # İstenen sayıya kırp
              return collob_rec
          else:
              return []

    else:
        return




# Kullanıcı için collobrative öneri al
user_id=int(user_info)


#collobrative filtreleme için fonksiyon
collob_rec = pearson_recommendations(user_id)




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''RECOMMENDATION VERİLERİNİ GÖNDER'''

if isinstance(collob_rec, pd.DataFrame) and not collob_rec.empty:
    for recommendation in collob_rec.to_dict('records'):
        movie_id, title = recommendation.get("movie_id"), recommendation.get("title")
        if movie_id is not None and title is not None:
            print("ID:", movie_id, "Title:", title)
            session.execute(text('INSERT INTO recommendations (show_id, title) VALUES (:show_id, :title)'),
                            {'show_id': movie_id, 'title': title})
            session.commit()
    print("İşlem tamamlandı.")
    connection.close()
elif isinstance(collob_rec, list) and collob_rec:
    for recommendation in collob_rec:
        movie_id, title = recommendation.get("movie_id"), recommendation.get("title")
        if movie_id is not None and title is not None:
            print("ID:", movie_id, "Title:", title)
            session.execute(text('INSERT INTO recommendations (show_id, title) VALUES (:show_id, :title)'),
                            {'show_id': movie_id, 'title': title})
            session.commit()
    print("İşlem tamamlandı.")
    connection.close()

else:
    print("Öneri bulunamadı.")
    connection.close()

if not clicked_last.empty:
    movie_for_recommend = clicked_last.iloc[0]['movieId']
    id = movie_for_recommend
    query_show_des = transformer_data.loc[transformer_data['movie_id'] == id]['description'].to_list()[0]
    if query_show_des:
        # query_show_des boş değilse devam et
        query_show_des = query_show_des[0]
        recommended_results = recommend(query_show_des)

        title_transformer_rec = []
        for index in recommended_results:
            if 0 <= index < len(transformer_data):
                movie_id_transformer_rec = transformer_data.iloc[index]['movie_id']
                title_transformer_rec.append(transformer_data.iloc[index]['title'])
                show_id = transformer_data.iloc[index]['movie_id']
                title = transformer_data.iloc[index]['title']
                print("ID:", show_id, "Title:", title)
                session.execute(text('INSERT INTO recommendations (show_id, title) VALUES (:show_id, :title)'),
                                {'show_id': int(show_id), 'title': str(title)})
                session.commit()
            else:
                print(f"Invalid index: {index}")


        print("işlem tamamlandı")
        connection.close()
    else:
        print("Kullanıcının tıkladığı film yok")
        connection.close()
else:
    print("Kullanıcı hiçbir filme tıklamadı")
    connection.close()

