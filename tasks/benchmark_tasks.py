"""
Benchmark tasks for evaluating planning strategies.
Three complexity levels: low, medium, high (6 tasks each = 18 total).
"""

BENCHMARK_TASKS = [
    # === LOW COMPLEXITY ===
    {"id": "low_01", "description": "What is the capital of France?",
     "complexity": "low", "domain": "factual_recall", "expected_answer": "Paris", "requires_tools": False},
    {"id": "low_02", "description": "Calculate: 15 * 8 + 12",
     "complexity": "low", "domain": "arithmetic", "expected_answer": "132", "requires_tools": True},
    {"id": "low_03", "description": "What is the largest planet in our solar system?",
     "complexity": "low", "domain": "factual_recall", "expected_answer": "Jupiter", "requires_tools": False},
    {"id": "low_04", "description": "What is the boiling point of water in Celsius?",
     "complexity": "low", "domain": "factual_recall", "expected_answer": "100", "requires_tools": False},
    {"id": "low_05", "description": "Calculate the square root of 144.",
     "complexity": "low", "domain": "arithmetic", "expected_answer": "12", "requires_tools": True},
    {"id": "low_06", "description": "Who created the Python programming language?",
     "complexity": "low", "domain": "factual_recall", "expected_answer": "Guido van Rossum", "requires_tools": False},

    # === MEDIUM COMPLEXITY ===
    {"id": "med_01", "description": "Look up the population of France and Germany, then tell me which country has a larger population and by how much (approximately in millions).",
     "complexity": "medium", "domain": "multi_step_qa", "expected_answer": "Germany has a larger population by approximately 16.7 million", "requires_tools": True},
    {"id": "med_02", "description": "Calculate: (25 * 4) + (sqrt(81) * 3). Show your work.",
     "complexity": "medium", "domain": "arithmetic", "expected_answer": "127", "requires_tools": True},
    {"id": "med_03", "description": "What continent is Japan in? Then look up Japan's population and calculate how many people per square kilometer Japan has, given its area is approximately 377,975 square kilometers.",
     "complexity": "medium", "domain": "multi_step_qa", "expected_answer": "326", "requires_tools": True},
    {"id": "med_04", "description": "Look up the GDP of the USA and Japan. Calculate what percentage Japan's GDP is of the USA's GDP.",
     "complexity": "medium", "domain": "multi_step_qa", "expected_answer": "15.3", "requires_tools": True},
    {"id": "med_05", "description": "Find the distance from Earth to the Moon and from Earth to the Sun. How many times farther is the Sun from Earth compared to the Moon?",
     "complexity": "medium", "domain": "multi_step_qa", "expected_answer": "389", "requires_tools": True},
    {"id": "med_06", "description": "Who invented the telephone and in what year? Then calculate how many years ago that was from 2024.",
     "complexity": "medium", "domain": "multi_step_qa", "expected_answer": "Alexander Graham Bell in 1876, which was 148 years ago", "requires_tools": True},

    # === HIGH COMPLEXITY ===
    {"id": "high_01", "description": "Compare the population density of France and Japan. Look up the population of each country and use these areas: France = 643,801 sq km, Japan = 377,975 sq km. Calculate the population density for each (people per sq km), determine which is more densely populated, and by what factor.",
     "complexity": "high", "domain": "multi_step_analysis", "expected_answer": "Japan is about 3.1 times more densely populated than France", "requires_tools": True},
    {"id": "high_02", "description": "A train travels from City A to City B at 60 km/h and returns at 40 km/h. The total distance for the round trip is 480 km. Step 1: Find the one-way distance. Step 2: Calculate the time for each leg of the journey. Step 3: Calculate the average speed for the entire round trip. Note: Average speed = total distance / total time, NOT the average of the two speeds.",
     "complexity": "high", "domain": "math_reasoning", "expected_answer": "48", "requires_tools": True},
    {"id": "high_03", "description": "Research the following: the GDP of USA, China, Japan, and Germany. Then: (1) Rank them from highest to lowest GDP, (2) Calculate the combined GDP of all four, (3) Calculate what percentage of the combined GDP each country represents. Present the results clearly.",
     "complexity": "high", "domain": "multi_step_analysis", "expected_answer": "USA 50.8%, China 33.1%, Germany 8.3%, Japan 7.8%", "requires_tools": True},
    {"id": "high_04", "description": "Solve this logic puzzle step by step: If the speed of light is approximately 300,000 km/s, and the distance from Earth to the Sun is about 149.6 million km, how many minutes does it take for sunlight to reach Earth? Then, if a spacecraft travels at 1/10th the speed of light, how many minutes would it take to reach the Sun from Earth?",
     "complexity": "high", "domain": "math_reasoning", "expected_answer": "Light takes about 8.3 minutes; spacecraft takes about 83 minutes", "requires_tools": True},
    {"id": "high_05", "description": "Consider this multi-step problem: A store has a 20% off sale. An item originally costs $150. Step 1: Calculate the sale price after the 20% discount. Step 2: Apply an additional 10% member discount on the sale price. Step 3: Add 8.5% sales tax to the final discounted price. Step 4: If the customer has a $15 coupon, what is the total amount they pay?",
     "complexity": "high", "domain": "math_reasoning", "expected_answer": "102.06", "requires_tools": True},
    {"id": "high_06", "description": "Compare the currencies of France, Japan, and the UK. Then look up each country's GDP. If 1 EUR = 1.10 USD, 1 JPY = 0.0067 USD, and 1 GBP = 1.27 USD, convert all GDPs to USD and rank the three countries. Which country has the highest GDP in USD terms?",
     "complexity": "high", "domain": "multi_step_analysis", "expected_answer": "Japan has the highest GDP among the three", "requires_tools": True},
]


def get_all_tasks():
    return BENCHMARK_TASKS

def get_tasks_by_complexity(complexity):
    return [t for t in BENCHMARK_TASKS if t["complexity"] == complexity]
