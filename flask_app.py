from flask import Flask, render_template, request


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)



class hello_world():

    def m(t):
        movie_name= t
        
        movies = pd.read_csv('movies.csv')

        selective_columns = ['genres', 'keywords','tagline', 'cast', 'director']


        for features in selective_columns:
            movies[features] = movies[features].fillna('')

        combine_features = movies['genres']+' '+movies['tagline'] + ' '+movies['cast']+' '+movies['director']

        vectorizer = TfidfVectorizer()
        feature_vectors= vectorizer.fit_transform(combine_features)

        similarity=cosine_similarity(feature_vectors)
        


        list_of_all_titled = movies['title'].tolist()


        find_close_match= difflib.get_close_matches(movie_name,list_of_all_titled)
        close_match=find_close_match[0]


        index_of_the_movie= movies[movies.title==close_match]['index'].values[0]


        similarity_score =list(enumerate(similarity[index_of_the_movie]))
        sorted_similarity_score=sorted(similarity_score,key= lambda x:x[1],reverse=True)


        i=1
        l=""
        for m in sorted_similarity_score:
            index = m[0]
            title_from_index =movies[movies.index==index]['title'].values[0]
            l=l+"\n"+title_from_index
            i+=1
            if i>10:
                break
        return l

@app.route('/', methods=['GET', 'POST'])
def input():
    a=""
    if request.method == 'POST':
        title= request.form['title']
        a=hello_world.m(title)
    
   
    return render_template('index1.html',a=a)
        



    

if __name__ == "__main__":
    app.run(debug=True)
