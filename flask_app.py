#from crypt import methods
from distutils.command.config import config
from flask import Flask, flash, redirect, url_for, render_template, request, Response
import pandas as pd
import os, io, csv, sys, pickle, time
from werkzeug.utils import secure_filename
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
        return redirect(url_for('character'))
        #return render_template('character_net.html')
    return render_template('index.html')



@app.route('/character_net', methods=['POST', 'GET'])
def character(**kwargs):

    book_dict = {}
  
    if request.method=='POST': 
        
        # for submitting the title and content
        if request.form['submit'] == "Submit Book Info":

            # test -----------------------------------------
            # check content
            if not request.files["book_content"]: 
                flash('Please provide the book content!')
                return redirect(request.url)


            book_content_file = request.files["book_content"]

            if book_content_file.filename == '':
                flash('No selected file!')
                return redirect(request.url)


            # check the name
            elif not allowed_file(book_content_file.filename): 
                flash('please provide a file with txt extention.') 
                return redirect(request.url)   
            
            # chapter name and title 
            for (k, v) in {'chapter': 'chapter_regex', 'title':'book_title'}.items():
                if not request.form[v]:
                    flash(f'Please provide the {k} description!')
                    return redirect(request.url)


            # if everything was ok
            if request.form['book_title'] and request.files['book_content']:

                # title 
                book_dict['book_title'] = request.form['book_title']

                # saving the content as a text file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'file_content.txt')
                book_content_file.save(file_path)

                with open(file_path, encoding="utf8") as f:
                    book_content = f.read()

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


        if request.form['submit'] == "Go with TransformerS!":
            
            from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
            classifier = pipeline('sentiment-analysis', device=0)



    else: return render_template('character_net.html')





# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files/'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.run(debug=True)

#------------------------------------------------------


