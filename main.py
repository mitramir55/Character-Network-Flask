#from crypt import methods
from distutils.command.config import config
import re
from cherrypy import url
from flask import Flask, flash, redirect, url_for, render_template, request, Response
from graphviz import render
import numpy as np
import pandas as pd
import os, io, csv, sys, pickle, time
from afinn import Afinn
from charset_normalizer import from_path
from itertools import chain 
from werkzeug.utils import secure_filename
from BookAnalyzer import Book_content_analyzer, Book_info_scraper
import plotly

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

def normalize_text(file_path):
    raw_file = from_path(file_path)
    book_content = str(raw_file.best())
    return book_content

def save_book_dict(book_dict):
    with open(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl', 'wb') as f:
        pickle.dump(book_dict, f)

def read_book_dict():
    book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
    return book_dict

# Flask ----------------------------------------------------------------
@app.route('/', methods=['POST', 'GET'])
def index():
    
    if request.method=='POST':
        return redirect(url_for('input_info'))
        
    return render_template('index.html')


@app.route('/input_info', methods=['POST', 'GET'])
def input_info(**kwargs):

    error = None
    book_dict = {}
  
    if request.method=='POST': 
        
        # for submitting the title and content
        if request.form['submit'] == "Submit Book Info":

            # test -----------------------------------------
            # check content
            if not request.files["book_content"]: 
                flash('Please provide the book content!')
                return redirect(url_for('input_info', error=True))

            book_content_file = request.files["book_content"]

            if book_content_file.filename == '':
                flash('No selected file!')
                return redirect(url_for('input_info', error=True))

            # check the name
            elif not allowed_file(book_content_file.filename): 
                flash('please provide a file with txt extention.') 
                return redirect(url_for('input_info', error=True))
            
            # check chapter name and title 
            for (k, v) in {'chapter': 'chapter_regex', 'title':'book_title'}.items():
                if not request.form[v]:
                    flash(f'Please provide the {k} description!')
                    return redirect(url_for('input_info', error=True))


            # if everything was ok
            if request.form['book_title'] and request.files['book_content']:

                book_dict['book_title'] = request.form['book_title']

                # saving the content as a text file
                filename = secure_filename(book_content_file.filename)
                file_path = save_file(book_content_file, filename)
                book_content = normalize_text(file_path)
                book_dict['book_content'] = book_content
                book_dict['chapter_regex']  = request.form["chapter_regex"]

                save_book_dict(book_dict)

                return redirect(url_for('book_info'))

    else: return render_template('input_info.html', error=error)

@app.route('/book_info', methods=['POST', 'GET'])
def book_info(**kwargs):
    if request.method=='POST': 
        return redirect(url_for('senti_analysis'))

    else: 
        book_dict = read_book_dict()

        # analysis --------------------------------
        analyzer = Book_content_analyzer() 
        book_content_cleaned = analyzer.clean_content(book_content=book_dict['book_content'], cu_patterns_to_remove = ['¡¡¡¡'])
        book_sentences = analyzer.spacy_detect_sentences(book_content_cleaned)
        finalized_sents = analyzer.clean_sentences(book_sentences, chapter_regex = book_dict['chapter_regex'])
        book_dict['finalized_sents'] = finalized_sents
        book_dict['number_of_sentences'] = len(finalized_sents)

        # scrape-----------------------------------
        book_scraper = Book_info_scraper()
        genres, reviews, ratings, author, year_published = book_scraper.get_goodreads_info(book_name=book_dict['book_title'])

        save_book_dict(book_dict)

        return render_template('book_info.html', genres=genres, reviews=reviews,
         ratings=ratings, author=author, year_published=year_published, length=book_dict['number_of_sentences'],
          book_title=book_dict['book_title'])


@app.route('/senti_analysis', methods=['POST', 'GET'])
def senti_analysis(**kwargs):

    if request.method=='POST': 

        # afinn ----------------------------------------------------------
        if request.form['submit'] == "Go with Afinn!":

            # analysis --------------------------------
            analyzer = Book_content_analyzer()
            book_dict = read_book_dict()
            sentiment_lables, encoded_sentiment_labels, emotions_count = analyzer.senti_analysis_Afinn(sentence_list=book_dict['finalized_sents'])        
            
            book_dict['sentiment_lables'] = sentiment_lables
            book_dict['encoded_sentiment_labels'] = encoded_sentiment_labels
            book_dict['emotions_count'] = emotions_count
            save_book_dict(book_dict)

            return render_template('senti_analysis.html',
                received=True, sentiment_lables=sentiment_lables, 
                encoded_sentiment_labels =encoded_sentiment_labels,
                emotions_count=emotions_count)

        # transformers ----------------------------------------------------------
        if request.form['submit'] == "Go with TransformerS!":

            flash("will be loaded after I get out of Iran!\n for now, we use previously processed files.")
            # unhash the following =======================================
            
            #sentiment_lables, encoded_sentiment_labels, emotions_count = analyzer.senti_analysis_transformers(book_dict['finalized_sents'])
            properties_folder_path = r"C:\Users\Lenovo\Flask apps\flask-app-character-net\Archive\sample results of analysis"
            book_content = pd.read_pickle(properties_folder_path + r'\book_content.pkl')
            sentiment_lables = pd.read_pickle(properties_folder_path + r'\sentiment_lables.pkl')
            encoded_sentiment_labels = pd.read_pickle(properties_folder_path + r'\emotions_count.pkl')
            emotions_count = pd.read_pickle(properties_folder_path + r'\emotions_count.pkl')
            
            save_book_dict(book_dict)

            return render_template('senti_analysis.html',
                received=True, sentiment_lables=sentiment_lables, 
                encoded_sentiment_labels =encoded_sentiment_labels,
                emotions_count=emotions_count)

        if request.form['submit'] == "Named Entity Recognition":
            return redirect(url_for('ner'))

    else: return render_template('senti_analysis.html')


@app.route('/ner', methods=['POST', 'GET'])
def ner(**kwargs):
    if request.method=='POST': 

        # afinn ----------------------------------------------------------
        if request.form['submit'] == "Find the Names!":

            # test--------------------
            if not request.form['n']:
                flash('Please specify the top n!')
                return redirect(url_for('ner', error=True))
                
            elif request.form['n']:

                book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
                n = int(request.form['n'])

                analyzer = Book_content_analyzer()
                sorted_names_dict = analyzer.find_most_pop_names(list_sents=book_dict['finalized_sents'])
                sorted_flatten_names_dict = analyzer.flatten_names(sorted_names_dict)

                book_dict['names_dict'] = sorted_flatten_names_dict
                book_dict['top_n'] = n

                save_book_dict(book_dict)

                df = pd.DataFrame({'Rank':i+1, 'Known as':k, 'Num. of Appearances':v} for i, (k,v) in enumerate(sorted_flatten_names_dict.items()))
                top_n_df = df.iloc[:n, :]

                

                return render_template('ner.html', length=len(df), names=top_n_df.loc[:, 'Known as'].values,
                 zip=zip, received=True, column_names=top_n_df.columns.values, row_data=list(top_n_df.values.tolist())) 


        if request.form['submit'] == 'Add and Remove These!':

            analyzer=Book_content_analyzer()
            missed_names = request.form['unrecognized_names']
            extra_names = request.form['extra_names']
            book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
            n = book_dict['top_n']

            names_dict = analyzer.add_or_remove_names(
                list_sents=book_dict['finalized_sents'],
                names_dict=book_dict['names_dict'], 
                extra_names=extra_names,
                missed_names=missed_names)

            book_dict['names_dict'] = names_dict


            save_book_dict(book_dict)


            sorted_flatten_names_dict = analyzer.flatten_names(names_dict)
            df = pd.DataFrame({'Rank':i, 'Known as':k, 'Num. of Appearances':v} for i, (k,v) in enumerate(sorted_flatten_names_dict.items()))
            top_n_df = df.iloc[:n, :]


            return render_template('ner.html', length=len(df), names=top_n_df.loc[:, 'Known as'].values, zip=zip, received=True,
            column_names=top_n_df.columns.values, row_data=list(top_n_df.values.tolist())) 


        if request.form['submit'] == "No problem! Go to the next step!":
            return redirect(url_for('cooccurance', received=None)) 



    else: return render_template('ner.html', received=None)


@app.route('/cooccurance', methods=['POST', 'GET'])
def cooccurance(**kwargs):

    if request.method=='POST': 
        if request.form['submit'] == "Give me the Cooccurrence!":
            
            book_dict = read_book_dict()
            analyzer = Book_content_analyzer()
            n = book_dict['top_n']
            top_n_popular_names = list(book_dict['names_dict'].keys())[:n]

            pop_names_df, cooccurrence_matrix, cooccurrence_matrix_with_senti = analyzer.create_cooccurrence_matrices(
                top_n_popular_names=top_n_popular_names, 
                book_sents=book_dict['finalized_sents'], 
                encoded_senti_labels=book_dict['encoded_sentiment_labels'],
                normalize_mode=True, threshold = 0)

            book_dict['pop_names_df'] = pop_names_df
            book_dict['cooccurrence_matrix'] = cooccurrence_matrix
            book_dict['cooccurrence_matrix_with_senti'] = cooccurrence_matrix_with_senti
            
            with open(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl', 'wb') as f:
                    pickle.dump(book_dict, f)

            # create dataframes from the matrices
            cooccurrence_df = pd.DataFrame(cooccurrence_matrix, columns=top_n_popular_names, index=top_n_popular_names).reset_index()
            cooccurrence_df_with_senti = pd.DataFrame(cooccurrence_matrix_with_senti, columns=top_n_popular_names, index=top_n_popular_names).reset_index()

            # create the index with characters names
            cooccurrence_df.rename(columns={'index':'Characters'}, inplace=True)
            cooccurrence_df_with_senti.rename(columns={'index':'Characters'}, inplace=True)


            return render_template('cooccurance.html', zip=zip, received=True,
                            column_names_1=cooccurrence_df.columns.values,
                            row_data_1=list(cooccurrence_df.values.tolist()),
                            column_names_2=cooccurrence_df_with_senti.columns.values,
                            row_data_2=list(cooccurrence_df_with_senti.values.tolist()),
                            )

        if request.form['submit'] == "See the progress of characters...":

            if not request.form['n_sections']:
                flash('Please enter the number of sections!', received=None)
                return render_template('cooccurance.html')

            if request.form['n_sections']:
                n_sections = request.form['n_sections']
                book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
                analyzer = Book_content_analyzer()
                n = book_dict['top_n']
                top_n_popular_names = list(book_dict['names_dict'].keys())[:n]

                graphJSON  = analyzer.create_plot_df(top_n_popular_names=top_n_popular_names,
                 pop_names_df=book_dict['pop_names_df'], n_sections=n_sections)
                return render_template('progress.html', graphJSON=graphJSON) 
        
    else: return render_template('cooccurance.html', received=None)

        
    
@app.route('/progress', methods=['POST', 'GET'])
def progress():

    if request.form['submit'] == 'Generate the Graph!':
        analyzer = Book_content_analyzer()
        book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
        n = book_dict['top_n']
        top_n_popular_names = list(book_dict['names_dict'].keys())[:n]


        graph_ = analyzer.matrix_to_edge(
            cooccurrence_matrix=book_dict['cooccurrence_matrix'],
            cooccurrence_matrix_with_senti=book_dict['cooccurrence_matrix_with_senti'],
            pop_names_df=book_dict['pop_names_df'], 
            top_n_popular_names=top_n_popular_names)

    else: return render_template('progress.html')



@app.route('/sent')
def sent():
    i=0
    m=0
    analyzer = Book_analyzer()
    book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')

    all_sentences = []
    list_corpus = [sent for sent in book_dict['book_content'].split('\n') if sent !='']
    length = len(list_corpus)
    hop = int(np.round(length/4)+1)

    def func (i):
        return render_template('progress_bar.html', progress=str(i)*0.25)

    for i in range(0, length, hop):
        all_sentences.append(analyzer.spacy_detect_sentences_editted(list_corpus=list_corpus[hop*i:hop*(i+1)]))
        func(i)

    all_sentences_final = list(chain.from_iterable(all_sentences[0]))
    return render_template('progress_bar.html', progress=100)



# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files/'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'

#------------------------------------------------------


