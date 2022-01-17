
from flask import Flask, flash, redirect, url_for, render_template, request
import pandas as pd
import os, io, csv
from werkzeug.utils import secure_filename


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Flask ----------------------------------------------------------------
app = Flask(__name__)


@app.route('/character_net', methods=['POST', 'GET'])
def character(mode='', **kwargs):

    # for submitting the title and content
    if request.method=='POST' and request.form.get("book_title"): 
        

        # content -----------------------------------------
        # check content
        book_content_file = request.files["book_content"]
        if book_content_file.filename == '':
            flash('No selected file!')

        # if everything was ok
        if book_content_file and allowed_file(book_content_file.filename):

            file_path = os.path.join(app.config['UPLOAD_FOLDER'] + 'file_content.txt')
            book_content_file.save(file_path)

            #book_content = pd.read_csv(file_path)
            with open(file_path, encoding="utf8") as f:
                book_content = f.read()

                
            # title ---------------------------------
            book_title = request.form['book_title']


        elif not allowed_file(book_content_file.filename): flash('please provide a file with txt extention.')    
        

            # analysis --------------------------------


            
        return render_template("character_net.html", received=True)


    return render_template('character_net.html')







# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files/'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.run(debug=True)

#------------------------------------------------------


