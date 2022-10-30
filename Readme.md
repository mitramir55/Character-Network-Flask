# Analyze your favourit Book!

This webapp is created to tell you what characters decide for the plot, who is the dominant one, and how they interact with each other. From November 2022, the website - [Link](https://book-character-network.herokuapp.com/) - can only be run locally since the host shut down its free servers and this app is too big for other free servers. 

*How to use*: clone the project, install Flask, and enter `flask run` in your terminal.

![preview of the website](https://github.com/mitramir55/flask-app-character-net/blob/master/static/preview%20pics/previewOfWebsite.png)

## A preview

We'll go through every step to give you an idea of what this app does.

### Book Info

First, you enter the book details (name and the way chapters are in the text). With the help of this info, the website scrapes more info from GoodReads website to tell you about the author, year of publishing, and other facts. The SpaCy small model is also utilized in this step to count the number of sentences in the file you upload.

![image](https://user-images.githubusercontent.com/53291220/198904178-61ab999c-320d-4dd4-b0d7-67d4a9036843.png)

### Sentiment Analysis

We separate sentences with SpaCy and analyze them with Afinn. At the end we get a nice graph of the overall mood of the book.

![pic2_senti](https://user-images.githubusercontent.com/53291220/198904193-e724f001-92ac-4ade-a33e-57a804a3de2d.png)


### Named Entity Recognition - NER

After saving the sentiment indicator for each sentence, we once again process them and look for names of people\characters in the book.

![pic3_ner1](https://user-images.githubusercontent.com/53291220/198904229-a65f1ec4-4213-4663-81c6-4a414dee80e4.png)

Sometimes, you might not see the names of specific characters or just find some redundant names between them. If there was any change you wanted to make in the list, you can simply use the last two boxes to write the names of people in.

![pic4_ner2](https://user-images.githubusercontent.com/53291220/198904237-2b16ba46-a225-4bc1-9d10-130e1d9f73df.png)

### Co-occurrence

For this part, we analyze at how characters interact with each other by looking at their mutual occurrence in one sentence. Then, we use the sentiment scores of these sentences and create a table indicating characters relationships.
 
![pic5_co1](https://user-images.githubusercontent.com/53291220/198904248-b07622e5-4b86-4e8b-b58b-28fafe7db28e.png)

### Character Progress

Because we want to get a good overview of characters' roles and importance, we also look at their presence through out the book. We do that by segmenting text into n pieces and counting the appearance of each person in that segment.

![pic6_progress](https://user-images.githubusercontent.com/53291220/198904265-9506f306-01e9-4450-a34f-8c27a694da86.png)

### The Graph

The final step is creating a graph. For this visualization, I utilized D3 library Graph force to depict nodes and edges.

![pic7_graph](https://user-images.githubusercontent.com/53291220/198904274-9660f1a6-e747-4cb7-9733-b5f6730b5f5e.png)

Resources:
[Observable graph force](https://observablehq.com/@d3/force-directed-graph)
Inspired by [Ken Huang's character network project](https://github.com/hzjken/character-network).

EPUB to txt convertor: [link](https://convertio.co/epub-txt/)



