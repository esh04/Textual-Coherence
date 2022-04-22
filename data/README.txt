This folder contains the train and test data for all four domains: Yahoo, Clinton, Enron, and Yelp.

Fields:
- text_id uniquely identifies each text
- question_title and question (Yahoo only): these were provided to annotators to give them additional context for the response, but we did not use them in training models
- subject (Clinton, Enron, Yelp): these were provided to annotators as additional context for the email, but were not used to train models
- text: the document itself
- ratingA1...ratingA3: 3 coherence ratings from expert annotators
- labelA: consensus label based on the 3 expert ratings
- ratingM1...ratingM5: 5 coherence ratings from MTurk annotators
- labelM: consensus label based on the 5 MTurk ratings

We have permission from Yahoo and Yelp to release this data.  We use the December 2017 version of the Yelp Dataset.  
