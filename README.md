# Text classification solution
The solution use transformer model for intent classification of ATIS dataset provided

## Model download
- Please download the intent_model from google drive as this is a big file
  https://drive.google.com/file/d/1yAuEs3d1lZ6xSXWVuR67ZyXhscDqVVHO/view?usp=sharing
- Please unzip and put it inside model package. 
- Then you can run the scripts which will pick up the files from there
- server.py can be run with the path to model



## Data preparation
- The data is multilabel separated by “+”. It was converted to a list of labels for each
sample
- An Enum of intents (all intents in training data) is used to make sure the api/classifier
supports a closed set of intents.
- Converted to numpy arrays before feeding to the NN.

## Train
- The training fine-tunes a pre-trained model from hugging face and its trainer to train.
“"distilbert-base-multilingual-cased" which is language agnostic.
- It only uses the training data and do 3-fold cross validation( ideally should do 10-fold)
and then train model on all train data and save the model
- To run training script, inside script
```json
PYTHONPATH=../ python train_intents.py
```
- The training uses gpu if available. I trained it in a AWS GPU instance and it took 2
hours for cross validation and building the final model.


##Eval
- To get evaluation on test data, inside script run
- PYTHONPATH=../ python test_eval.py
- It will print a classification report with an f-score for each intent.
- You can find the results already in results/test_eval_report.txt


## Server
- It uses Flask to serve the model and follows the spec specified in the challenge.
- It handles the errors properly and send response accordingly.
- To run the server, run
```json
python server.py --model model/intent_model.bin
```

##Notes

- I wrote a small test for the classifier to showcase how the code should be tested.
Ideally all methods in bert_intent_classifier should be tested. And API responses for
different cases should be tested as well. I didn’t do it for lack of time.
- Usually we use Dockerfile and Makefile to run everything in a container with simple
commands. I kept it simple on this one because of time constraints.
- For experiments, I have seen that pre-trained transformer models for text
classification works really well with a low number of samples. I usually try simpler
models with tf-idf features as it’s faster to train but since this was more a NN
challenge I didn’t write code for those experiments.


# Ultimate - ML Engineer Coding Challenge
We're excited that you want to join the Ultimate team.  If you have any questions regarding this task, please don't hesitate to ask.

## Brief
Your task is to implement a neural network-based intent classifier that can be used to provide inferencing service via an HTTP Service. The boiler plate for the Service is implemented in file `server.py` and you'll have to implement the API function for inferencing as per the API documentation provided below. The neural network interface has been defined in `intent_classifer.py`. You can add any methods and functionality to this class you deem necessary.

You may use any deep learning library (Tensorflow, Keras, PyTorch, ...) you wish and you can also use pre-existing components for building the network architecture if these would be useful in real-life production systems. Provide tooling and instructions for training the network from scratch.

## Implementation Notes / Requirements
- ATIS data can be used for training and developing the network. You'll find the data files in `data/atis` directory. Files are TSV files where the first column is the text and the second column is the intent label. ATIS data is in English only but extra points are given for language-agnostic implementation.
- The given codebase contains one bug (that we know of). You need to find and fix this bug.
- Your service needs to adopt the following API Documentation.


## API Documentation
API documentation for intent classification service.

### `GET /ready`
Returns HTTP status code 200 with response body `"OK"` when the server is running, model has been loaded and is ready to
serve infer requests and 423 with response body `"Not ready"` when the model has not been loaded.

### `POST /intent`
Responds intent classification results for the given query utterance.

#### Request
JSON request with MIME-Type of `application/json` and body:
- **text** `string`: Input sentence intent classification

Example request
```json
{
  "text": "find me a flight that flies from memphis to tacoma"
}
```

#### Response
JSON response with body:
- **intents** `[Prediction]`: An array of top 3 intent prediction results. See `Prediction` type below.

`Prediction` is a JSON object with fields:
- **label** `string`: Intent label name
- **confidence** `float`: Probability for the predicted intent

Example response
```json
{
  "intents": [{
    "label": "flight",
    "confidence": 0.73
  }, {
    "label": "aircraft",
    "confidence": 0.12
  }, {
    "label": "capacity",
    "confidence": 0.03
  }]
}
```

#### Exceptions
All exceptions are JSON repsonses with HTTP status code other than 2XX, error label and human readable error message.

##### 400 Body missing
Given when the request is missing a body.
```json
{
  "label": "BODY_MISSING",
  "message": "Request doesn't have a body."
}
```

##### 400 Text missing
Given when the request has body but the body is missing text field.
```json
{
  "label": "TEXT_MISSING",
  "message": "\"text\" missing from request body."
}
```

##### 400 Invalid text type
Given when the text field in the body is not string.
```json
{
  "label": "INVALID_TYPE",
  "message": "\"text\" is not a string."
}
```

##### 400 Text is empty
Given when the text in the body is an empty string.
```json
{
  "label": "TEXT_EMPTY",
  "message": "\"text\" is empty."
}
```

##### 500 Internal error
Given with any other exception. Human readable message includes the exception text.
```json
{
  "label": "INTERNAL_ERROR",
  "message": "<ERROR_MESSAGE>"
}
```

## Evaluation
- **Scenario fitness:** How does your solution meet the requirements?
- **Modularity:** Can your code easily be modified? How much effort is needed to add a new kind of ML model to your inference service?
- **Research & Experimentation:** What kind of experiments you did to select best model and features?
- **Code readability and comments:** Is your code easily comprehensible and testable?
- **Bonus:** Any additional creative features: Docker files, architectural diagrams for model or service, Swagger, model performance metrics etc. 
