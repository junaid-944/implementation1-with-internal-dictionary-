"""
Simulated tools that agents can use during task execution.
These provide deterministic behavior for reproducible benchmarking.
"""
import math

KNOWLEDGE_BASE = {
    "population_france": "The population of France is approximately 67.75 million (2024).",
    "population_germany": "The population of Germany is approximately 84.48 million (2024).",
    "population_uk": "The population of the United Kingdom is approximately 67.74 million (2024).",
    "population_japan": "The population of Japan is approximately 123.3 million (2024).",
    "population_usa": "The population of the United States is approximately 334 million (2024).",
    "capital_france": "The capital of France is Paris.",
    "capital_germany": "The capital of Germany is Berlin.",
    "capital_japan": "The capital of Japan is Tokyo.",
    "capital_uk": "The capital of the United Kingdom is London.",
    "capital_usa": "The capital of the United States is Washington, D.C.",
    "gdp_usa": "The GDP of the United States is approximately $27.36 trillion (2024).",
    "gdp_china": "The GDP of China is approximately $17.79 trillion (2024).",
    "gdp_japan": "The GDP of Japan is approximately $4.19 trillion (2024).",
    "gdp_germany": "The GDP of Germany is approximately $4.46 trillion (2024).",
    "gdp_france": "The GDP of France is approximately $3.13 trillion (2024).",
    "president_usa": "The president of the United States is Joe Biden (as of 2024).",
    "prime_minister_uk": "The Prime Minister of the UK is Keir Starmer (as of 2024).",
    "language_france": "The official language of France is French.",
    "language_germany": "The official language of Germany is German.",
    "language_japan": "The official language of Japan is Japanese.",
    "continent_france": "France is located in Europe.",
    "continent_japan": "Japan is located in Asia.",
    "continent_usa": "The United States is located in North America.",
    "area_france": "The area of France is approximately 643,801 square kilometers.",
    "area_usa": "The area of the United States is approximately 9,833,520 square kilometers.",
    "area_japan": "The area of Japan is approximately 377,975 square kilometers.",
    "currency_usa": "The currency of the United States is the US Dollar (USD).",
    "currency_japan": "The currency of Japan is the Japanese Yen (JPY).",
    "currency_uk": "The currency of the United Kingdom is the Pound Sterling (GBP).",
    "currency_france": "The currency of France is the Euro (EUR).",
    "speed_of_light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "boiling_point_water": "The boiling point of water at standard atmospheric pressure is 100 degrees Celsius.",
    "planets_solar_system": "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    "largest_planet": "Jupiter is the largest planet in our solar system.",
    "distance_earth_moon": "The average distance from the Earth to the Moon is about 384,400 kilometers.",
    "distance_earth_sun": "The average distance from the Earth to the Sun is about 149.6 million kilometers.",
    "python_creator": "Python was created by Guido van Rossum and first released in 1991.",
    "inventor_telephone": "Alexander Graham Bell is credited with inventing the telephone in 1876.",
    "inventor_lightbulb": "Thomas Edison is credited with inventing the practical incandescent light bulb in 1879.",
    "year_moon_landing": "The first Moon landing was on July 20, 1969, during the Apollo 11 mission.",
    "elements_water": "Water (H2O) is composed of two hydrogen atoms and one oxygen atom.",
    "largest_ocean": "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "tallest_mountain": "Mount Everest is the tallest mountain above sea level at 8,849 meters.",
    "longest_river": "The Nile River, at approximately 6,650 km, is often considered the longest river in the world.",
}


def calculator(expression: str) -> str:
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow, "log": math.log,
            "log10": math.log10, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        expr = expression.replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


def search_knowledge(query: str) -> str:
    query_lower = query.lower()
    best_match = None
    best_score = 0
    for key, value in KNOWLEDGE_BASE.items():
        key_words = key.replace("_", " ").split()
        score = sum(1 for word in key_words if word in query_lower)
        value_words = value.lower().split()
        score += sum(0.5 for word in query_lower.split() if word in " ".join(value_words))
        if score > best_score:
            best_score = score
            best_match = value
    if best_match and best_score > 0:
        return best_match
    return "No results found for the given query."


def lookup_fact(topic: str) -> str:
    topic_lower = topic.lower().replace(" ", "_")
    if topic_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[topic_lower]
    for key, value in KNOWLEDGE_BASE.items():
        if topic_lower in key or key in topic_lower:
            return value
    for key, value in KNOWLEDGE_BASE.items():
        key_parts = key.split("_")
        topic_parts = topic_lower.split("_")
        if any(p in key_parts for p in topic_parts):
            return value
    return f"No fact found for topic: {topic}"


def get_all_tools() -> dict:
    return {
        "calculator": {
            "function": calculator,
            "description": "Evaluate a mathematical expression. Input: a math expression string (e.g., '2 + 3 * 4'). Output: the numerical result.",
        },
        "search_knowledge": {
            "function": search_knowledge,
            "description": "Search for information on a topic. Input: a natural language query (e.g., 'population of France'). Output: relevant factual information.",
        },
        "lookup_fact": {
            "function": lookup_fact,
            "description": "Look up a specific fact by topic name. Input: a topic string (e.g., 'capital_france'). Output: the factual answer.",
        },
    }
