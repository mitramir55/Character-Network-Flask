EPUB to txt convertor: [link](https://convertio.co/epub-txt/)

how to add a plot in the senti analysis
how to create book info
can the number of characters be identified automatically?
get the endcoding of the txt file from the user to prevent â\x80\x99s.......

https://pypi.org/project/charset-normalizer/

Add the naive Bayes analyzer 
```
# Applying the NaiveBayesAnalyzer
blob_object = TextBlob('mitra is greatly bad', analyzer=NaiveBayesAnalyzer())
print(blob_object.sentiment)
```