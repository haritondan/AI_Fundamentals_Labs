from production import forward_chain_with_questions, backward_chain_with_questions
from rules import NINJA_RULES, NINJA_DATA

def interactive_expert_system():
    """
    The main function for interacting with the user to perform forward or backward chaining.
    """

    print("Welcome to the Luna-City Tourist Detection Expert System!")
    
    known_facts = set()

    # Ask the user to choose between forward or backward chaining
    mode = input("Choose mode: 'forward' or 'backward': ").strip().lower()

    if mode == 'f':
        print("\nYou chose forward chaining.")
        # Perform forward chaining using dynamically generated questions
        forward_chain_with_questions(NINJA_RULES, known_facts, verbose=True)
    elif mode == 'b':
        print("\nYou chose backward chaining.")
        hypothesis = input("Please provide a hypothesis (e.g., 'Shikamaru a Chunin'): ").strip()
        backward_chain_with_questions(NINJA_RULES, hypothesis, known_facts=NINJA_DATA, verbose=True)

    else:
        print("Invalid mode. Please restart the system and choose 'forward' or 'backward'.")

if __name__ == '__main__':
    while True:
        interactive_expert_system()
