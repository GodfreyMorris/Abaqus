import random

# Define ranges and options for each tire design parameter
CROWN_RADIUS_RANGE1 = (75, 90)
CROWN_RADIUS_RANGE2 = (80, 100)
CROWN_RADIUS_RANGE3 = (115, 125)
SIDEWALL_RADIUS_RANGE1 = (115, 125)
SIDEWALL_RADIUS_RANGE2 = (115, 125)

# Define scaling bounds
SCALING_BOUND_X = (0.8, 1.05)
SCALING_BOUND_Y = (0.9, 1.15)

# Function to generate a random chromosome
def generate_chromosome():
    chromosome = {
        'Crown Radius C1': random.uniform(*CROWN_RADIUS_RANGE1),
        'Crown Radius C2': random.uniform(*CROWN_RADIUS_RANGE2),
        'Crown Radius C3': random.uniform(*CROWN_RADIUS_RANGE3),
        'Sidewall Radius S1': random.uniform(*SIDEWALL_RADIUS_RANGE1),
        'Sidewall Radius S2': random.uniform(*SIDEWALL_RADIUS_RANGE2),
    }
    return chromosome

# Generate an initial population of chromosomes
def generate_initial_population(population_size):
    population = [generate_chromosome() for _ in range(population_size)]
    print("\nInitial Population:")
    for i, individual in enumerate(population):
        print(f"Individual {i+1}: {individual}")
    return population

# Example fitness function (placeholder)
def fitness_function(chromosome):
    # Define a fitness function that evaluates the tire design
    # Here, a simple placeholder function is used
    # Replace this with a proper fitness function based on design criteria
    fitness = random.uniform(0, 1)
    return fitness

# Evaluate the population
def evaluate_population(population):
    fitnesses = [fitness_function(chromosome) for chromosome in population]
    print("\nFitness Evaluations:")
    for i, fitness in enumerate(fitnesses):
        print(f"Individual {i+1} Fitness: {fitness:.4f}")
    return fitnesses

# Selection: Choose the top N individuals based on fitness
def select_parents(population, fitnesses, num_parents):
    parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    selected_parents = [parent for parent, fitness in parents[:num_parents]]
    print("\nSelected Parents:")
    for i, parent in enumerate(selected_parents):
        print(f"Parent {i+1}: {parent}")
    return selected_parents

# Crossover: Single-point crossover for simplicity
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = {**parent1, **dict(list(parent2.items())[crossover_point:])}
    child2 = {**parent2, **dict(list(parent1.items())[crossover_point:])}
    print("\nCrossover Result:")
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    print(f"Child 1: {child1}")
    print(f"Child 2: {child2}")
    return child1, child2

# Mutation: Randomly mutate a parameter in the chromosome
def mutate(chromosome):
    parameter_to_mutate = random.choice(list(chromosome.keys()))
    if 'Crown Radius' in parameter_to_mutate:
        scaling_factor = random.uniform(*SCALING_BOUND_X)
    elif 'Sidewall Radius' in parameter_to_mutate:
        scaling_factor = random.uniform(*SCALING_BOUND_Y)
    
    chromosome[parameter_to_mutate] *= scaling_factor
    
    # Ensure the parameter stays within its original range
    if 'Crown Radius' in parameter_to_mutate:
        chromosome[parameter_to_mutate] = min(max(chromosome[parameter_to_mutate], 115), 125)
    elif 'Sidewall Radius' in parameter_to_mutate:
        chromosome[parameter_to_mutate] = min(max(chromosome[parameter_to_mutate], 115), 125)
    
    print("\nMutation Result:")
    print(f"Mutated Chromosome: {chromosome}")
    return chromosome

# Main genetic algorithm loop
def genetic_algorithm(population_size, num_generations, num_parents, mutation_rate):
    # Generate initial population
    population = generate_initial_population(population_size)

    for generation in range(num_generations):
        print(f"\nGeneration {generation + 1}")

        # Evaluate the population
        fitnesses = evaluate_population(population)

        # Select parents
        parents = select_parents(population, fitnesses, num_parents)

        # Create next generation
        next_generation = []

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])

        # Mutate children
        for i in range(len(next_generation)):
            if random.random() < mutation_rate:
                next_generation[i] = mutate(next_generation[i])

        # Ensure the next generation doesn't exceed the population size
        next_generation = next_generation[:population_size]

        # Replace old population with new generation
        population = next_generation

    # Return the best individual from the final generation
    final_fitnesses = evaluate_population(population)
    best_individual = population[final_fitnesses.index(max(final_fitnesses))]

    best_fitness = max(final_fitnesses)
    print(f"\nBest Fitness: {best_fitness}")
    print("Best Tire Design:", best_individual)

    return best_individual

# Parameters
population_size = 20
num_generations = 5  # Reduced for brevity
num_parents = 4
mutation_rate = 0.1

# Run the genetic algorithm
best_tire_design = genetic_algorithm(population_size, num_generations, num_parents, mutation_rate)
print("\nBest Tire Design:", best_tire_design)