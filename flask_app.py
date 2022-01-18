#from crypt import methods
from flask import Flask, flash, redirect, url_for, render_template, request
import pandas as pd
import os, io, csv, sys
from werkzeug.utils import secure_filename
sys.path.append(r'C:\Users\Lenovo\character-network')

from character_net_src import Book_analyzer

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Flask ----------------------------------------------------------------
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    
    if request.method=='POST':
        return render_template('character_net.html')
    return render_template('index.html')



@app.route('/character_net', methods=['POST', 'GET'])
def character(**kwargs):

    # for submitting the title and content
    if request.method=='POST' and request.form.get("book_title"): 
        
        received=False

        # content -----------------------------------------
        # check content
        book_content_file = request.files["book_content"]
        if book_content_file.filename == '':
            flash('No selected file!')

        # check the name
        elif not allowed_file(book_content_file.filename): flash('please provide a file with txt extention.')    
        
        # if everything was ok
        elif book_content_file and allowed_file(book_content_file.filename):


            file_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'file_content.txt')
            book_content_file.save(file_path)

            #book_content = pd.read_csv(file_path)
            with open(file_path, encoding="utf8") as f:
                book_content = f.read()

                
            # title ---------------------------------
            book_title = request.form['book_title']

            # analysis --------------------------------
            analyzer = Book_analyzer()

            # ask the user to add to this  cu_patterns_to_remove 
            book_content_cleaned = analyzer.clean_content(book_content=book_content, cu_patterns_to_remove = ['¡¡¡¡'])

            # detect sents
            book_sentences = analyzer.spacy_detect_sentences(book_content_cleaned)

            return render_template("character_net.html", received=True)
        
        # ask for chapter regex
        if request.method=='POST' and request.form.get("chapter_regex"):

            chapter_regex = request.form["chapter_regex"]

            if chapter_regex!= 'no chapter':
                finalized_sents = analyzer.clean_sentences(book_sentences, chapter_regex = chapter_regex)

            else:
                finalized_sents = analyzer.clean_sentences(book_sentences, chapter_regex = 'No chapter')

            
            length = len(finalized_sents)
                
            return render_template("character_net.html", received=True, length=length)




            


    else: return render_template('character_net.html')







# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files/'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.run(debug=True)

#------------------------------------------------------


