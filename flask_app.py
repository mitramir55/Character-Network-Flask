from flask import Flask, flash, redirect, url_for, render_template, request
import pandas as pd
import os, io, csv




# Flask ----------------------------------------------------------------
app = Flask(__name__)


@app.route('/character_net', methods=['POST', 'GET'])
def character(mode='', **kwargs):

    return render_template('character_net.html')







# config------------------------------------------------
UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.run(debug=True)

#------------------------------------------------------


