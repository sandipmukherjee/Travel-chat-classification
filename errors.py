class InternalServerError(Exception):
    pass


class APIException(Exception):
    code = 400
    label = "API_ERROR"
    message = "Api Error"


class BODYMissingException(APIException):
    code = 400
    label = "BODY_MISSING"
    message = "Request doesn't have a body."


class EmptyTextException(APIException):
    code = 400
    label = "TEXT_MISSING"
    message = '"text" missing from request body.'


class NonAlphabeticTextException(APIException):
    code = 400
    label = "INVALID_TYPE"
    message = '"text" is not a string.'

