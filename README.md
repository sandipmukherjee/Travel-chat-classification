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
