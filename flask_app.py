#from crypt import methods
from distutils.command.config import config
from flask import Flask, flash, redirect, url_for, render_template, request, Response
import numpy as np
import pandas as pd
import os, io, csv, sys, pickle, time
from werkzeug.utils import secure_filename
from afinn import Afinn
from charset_normalizer import from_path


sys.path.append(r'C:\Users\Lenovo\character-network')
pd.set_option('display.float_format','{:.2f}'.format)


from character_net_src import Book_analyzer

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Flask ----------------------------------------------------------------
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    
    if request.method=='POST':
        return redirect(url_for('character_net'))
        
    return render_template('index.html')



@app.route('/character_net', methods=['POST', 'GET'])
def character_net(**kwargs):

    error = None
    book_dict = {}
  
    if request.method=='POST': 
        
        # for submitting the title and content
        if request.form['submit'] == "Submit Book Info":

            # test -----------------------------------------
            # check content
            if not request.files["book_content"]: 
                flash('Please provide the book content!')
                return redirect(url_for('character_net', error=True))


            book_content_file = request.files["book_content"]

            if book_content_file.filename == '':
                flash('No selected file!')
                return redirect(url_for('character_net', error=True))


            # check the name
            elif not allowed_file(book_content_file.filename): 
                flash('please provide a file with txt extention.') 
                return redirect(url_for('character_net', error=True))
            
            # chapter name and title 
            for (k, v) in {'chapter': 'chapter_regex', 'title':'book_title'}.items():
                if not request.form[v]:
                    flash(f'Please provide the {k} description!')
                    return redirect(url_for('character_net', error=True))


            # if everything was ok
            if request.form['book_title'] and request.files['book_content']:

                # title 
                book_dict['book_title'] = request.form['book_title']

                # saving the content as a text file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'file_content.txt')
                book_content_file.save(file_path)

                results = from_path(file_path)
                book_content = str(results.best())

                # store the content in the dict
                book_dict['book_content'] = book_content

                chapter_regex = request.form["chapter_regex"]
                
                book_dict['chapter_regex'] = chapter_regex

                # analysis --------------------------------
                analyzer = Book_analyzer()

                # ask the user to add to this  cu_patterns_to_remove 
                book_content_cleaned = analyzer.clean_content(book_content=book_content, cu_patterns_to_remove = ['¡¡¡¡'])
                
                
                #here should be the progress bar ===========================================
                    
                # detect sents
                book_sentences = analyzer.spacy_detect_sentences(book_content_cleaned)

                # clean the sentences
                finalized_sents = analyzer.clean_sentences(book_sentences, chapter_regex = chapter_regex)

                
                book_dict['finalized_sents'] = finalized_sents

                length = len(finalized_sents)
                book_dict['number_of_sentences'] = length


                with open(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl', 'wb') as f:
                    pickle.dump(book_dict, f)

                return render_template('character_net.html', received=True, length=length)

        if request.form['submit'] == "Go to Sentiment Analysis!":

            return redirect(url_for('senti_analysis'))

        if request.form['submit'] == "Go to Named Entity recognition!":

            return redirect(url_for('senti_analysis'))


    else: return render_template('character_net.html', error=error)

    
@app.route('/senti_analysis', methods=['POST', 'GET'])
def senti_analysis(**kwargs):

    
    if request.method=='POST': 

        # afinn ----------------------------------------------------------
        if request.form['submit'] == "Go with Afinn!":

            # analysis --------------------------------
            analyzer = Book_analyzer()
            book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
            sentiment_lables, encoded_sentiment_labels, emotions_count = analyzer.senti_analysis_Afinn(sentence_list=book_dict['finalized_sents'])        
            
            book_dict['sentiment_lables'] = sentiment_lables
            book_dict['encoded_sentiment_labels'] = encoded_sentiment_labels
            book_dict['emotions_count'] = emotions_count

            
            with open(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl', 'wb') as f:
                pickle.dump(book_dict, f)


            return render_template('senti_analysis.html',
                received=True, sentiment_lables=sentiment_lables, 
                encoded_sentiment_labels =encoded_sentiment_labels,
                emotions_count=emotions_count)

        # transformers ----------------------------------------------------------
        if request.form['submit'] == "Go with TransformerS!":

            flash("will be loaded after I get out of Iran!\n for now, import the sentiment file.")
            # unhash the following =======================================
            
            #sentiment_lables, encoded_sentiment_labels, emotions_count = analyzer.senti_analysis_transformers(book_dict['finalized_sents'])

            book_content = pd.read_pickle(r'C:\Users\Lenovo\flask-app-character-net\first_book_props\book_content.pkl')
            sentiment_lables = pd.read_pickle(r'C:\Users\Lenovo\flask-app-character-net\first_book_props\sentiment_lables.pkl')
            encoded_sentiment_labels = pd.read_pickle(r'C:\Users\Lenovo\flask-app-character-net\first_book_props\emotions_count.pkl')
            emotions_count = pd.read_pickle(r'C:\Users\Lenovo\flask-app-character-net\first_book_props\emotions_count.pkl')

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

                analyzer = Book_analyzer()
                sorted_names_dict = analyzer.find_most_pop_names(list_sents=book_dict['finalized_sents'])
                sorted_flatten_names_dict = analyzer.flatten_names(sorted_names_dict)

                book_dict['names_dict'] = sorted_flatten_names_dict
                book_dict['top_n'] = n


                with open(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl', 'wb') as f:
                    pickle.dump(book_dict, f)


                df = pd.DataFrame({'Rank':i+1, 'Known as':k, 'Num. of Appearances':v} for i, (k,v) in enumerate(sorted_flatten_names_dict.items()))
                top_n_df = df.iloc[:n, :]

                length=len(df)

                return render_template('ner.html', length=length, top_n_popular_names=top_n_df, zip=zip, received=True,
                    column_names=top_n_df.columns.values, row_data=list(top_n_df.values.tolist())) 


        if request.form['submit'] == 'Add and Remove These!':

            analyzer=Book_analyzer()
            missed_names = request.form['unrecognized_names']
            extra_names = request.form['extra_names']
            book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
            n = book_dict['top_n']

            names_dict = analyzer.add_or_remove_names(
                list_sents=book_dict['finalized_sents'],
                names_dict=book_dict['names_dict'], 
                extra_names=extra_names,
                missed_names=missed_names)


            sorted_flatten_names_dict = analyzer.flatten_names(names_dict)
            df = pd.DataFrame({'Rank':i, 'Known as':k, 'Num. of Appearances':v} for i, (k,v) in enumerate(sorted_flatten_names_dict.items()))
            top_n_df = df.iloc[:n, :]


            return render_template('ner.html', length=len(df), top_n_popular_names=top_n_df, zip=zip, received=True,
            column_names=top_n_df.columns.values, row_data=list(top_n_df.values.tolist())) 


        if request.form['submit'] == "No problem! Go to the next step!":

            return redirect(url_for('cooccurance', received=None)) 



    else: return render_template('ner.html', received=None)


@app.route('/cooccurance', methods=['POST', 'GET'])
def cooccurance(**kwargs):

    if request.method=='POST': 
        if request.form['submit'] == "Give me the Cooccurrence!":
            
            book_dict = pd.read_pickle(app.config['UPLOAD_FOLDER'] + 'book_dict.pkl')
            analyzer = Book_analyzer()
            n = book_dict['top_n']
            top_n_popular_names = list(book_dict['names_dict'].keys())[:n]

            pop_names_df, cooccurrence_matrix, cooccurrence_matrix_with_senti = analyzer.create_cooccurrence_matrices(
                top_n_popular_names=top_n_popular_names, 
                book_sents=book_dict['finalized_sents'], 
                encoded_senti_labels=book_dict['encoded_sentiment_labels'],
                normalize_mode=True, threshold = 2)

            cooccurrence_df = pd.DataFrame(cooccurrence_matrix, columns=top_n_popular_names, index=top_n_popular_names)
            cooccurrence_df_with_senti = pd.DataFrame(cooccurrence_matrix_with_senti, columns=top_n_popular_names, index=top_n_popular_names)

            return render_template('cooccurance.html', zip=zip, received=True,
                            column_names_1=cooccurrence_df.columns.values,
                            row_data_1=list(cooccurrence_df.values.tolist()),
                            column_names_2=cooccurrence_df_with_senti.columns.values,
                            row_data_2=list(cooccurrence_df_with_senti.values.tolist()),
                            )

        
    else: return render_template('cooccurance.html', received=None)

        
    






# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files/'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'

if __name__ == "__main__":
    app.run(debug=True)

#------------------------------------------------------


