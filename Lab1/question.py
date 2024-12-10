import random

def generate_multiple_choice_question(subject):
    print(f"Choose one of the following traits about {subject}:")
    choices = [
        f'{subject} has no skill', f'{subject} has stealthy movements', 
        f'{subject} has a team leader', f'{subject} has nature link', 
        f'{subject} has multiple jutsu', f'{subject} has written book'
    ]
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    
    choice_index = int(input("Enter the number of your choice: ")) - 1
    return choices[choice_index]

def generate_open_ended_question(subject, consequent):
    return f"Tell us more about how {consequent.replace('(?x)', subject, 1)}?"

def generate_yes_no_question(subject, antecedent):
    return f"Does {antecedent.replace('(?x)', subject, 1)}? (yes/no)"


def generate_backward_multiple_choice_question(subject, correct_antecedent, all_antecedents):
    """
    This will present one correct option and a few distractions.
    
    Args:
        - subject: The subject of the hypothesis (e.g., 'Naruto').
        - correct_antecedent: The correct antecedent we want the user to pick.
        - all_antecedents: All antecedents of the rule to generate distractions.
    
    Returns:
        - The correct antecedent selected by the user.
    """
    correct_option = correct_antecedent.replace('(?x)', subject)

    # Generate distraction options by randomly picking other traits that are not the correct antecedent
    distractions = [f'{subject} has stealthy movements', f'{subject} has no skill', f'{subject} has nature link', f'{subject} has written book']
    distractions = [d.replace('(?x)', subject) for d in distractions if d != correct_option]
    distractions = random.sample(distractions, min(3, len(distractions)))  # Choose 3 or fewer distractions

    # Combine the correct option with distractions and shuffle them
    options = distractions + [correct_option]
    random.shuffle(options)

    # Present the multiple-choice question
    print(f"Choose one of the following traits about {subject}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    # Get the user's choice
    choice_index = int(input("Enter the number of your choice: ")) - 1
    return options[choice_index] == correct_option  # Return True if the correct option is chosen
