from enum import unique, Enum


@unique
class Intent(str, Enum):
    CAPACITY = "capacity"
    AIRPORT = "airport"
    CHEAPEST = "cheapest"
    RESTRICTION = "restriction"
    MEAL = "meal"
    CITY = "city"
    AIRCRAFT = "aircraft"
    AIRFARE = "airfare"
    AIRLINE = "airline"
    DISTANCE = "distance"
    FLIGHT = "flight"
    QUANTITY = "quantity"
    GROUND_SERVICE = "ground_service"
    ABBREVIATION = "abbreviation"
    GROUND_FARE = "ground_fare"
    FLIGHT_TIME = "flight_time"
    FLIGHT_NO = "flight_no"
