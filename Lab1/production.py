import re
from utils import *
from question import generate_multiple_choice_question, generate_open_ended_question, generate_yes_no_question,  generate_backward_multiple_choice_question

### We've tried to keep the functions you will need for
### back-chaining at the top of this file. Keep in mind that you
### can get at this documentation from a Python prompt:
###
### >>> import production
### >>> help(production)



def forward_chain_with_questions(rules, data, verbose=False):
    """
    Forward chaining with dynamic questions to gather facts.
    """
    # Ask for the subject of the questions (character's name)
    subject = input("Who are we asking about? (e.g., Naruto, Shikamaru): ").strip()

    # Step 1: Ask a multiple-choice question to gather the first fact
    fact = generate_multiple_choice_question(subject)
    data.add(fact)
    
    # Step 2: Find the rule that contains the confirmed first fact
    for rule in rules:
        antecedents = rule.antecedent()
        if fact in [antecedent.replace('(?x)', subject, 1) for antecedent in antecedents]:
            # We found a rule containing the confirmed fact, now ask about the second antecedent
            other_fact = [antecedent for antecedent in antecedents if antecedent.replace('(?x)', subject, 1) != fact][0]
            open_ended_question = generate_open_ended_question(subject, rule.consequent()[0])
            print(open_ended_question)
            
            # Step 3: Ask an open-ended question to get more details
            answer = input(f"Provide more details about {subject}:").strip().lower()
            if other_fact.replace('(?x)', subject, 1) in answer:
                data.add(other_fact.replace('(?x)', subject, 1))
                
                # Step 4: Find the rule where the consequent becomes the antecedent for the next rule
                consequent = rule.consequent()[0].replace('(?x)', subject, 1)
                for next_rule in rules:
                    if consequent in [antecedent.replace('(?x)', subject, 1) for antecedent in next_rule.antecedent()]:
                        # Step 5: Confirm the tourist type using a yes/no question
                        next_antecedent = next_rule.antecedent()[0].replace('(?x)', subject, 1)
                        yes_no_question = generate_yes_no_question(subject, next_antecedent)
                        confirmation = input(yes_no_question).strip().lower()
                        if confirmation == 'yes':
                            tourist_type = next_rule.consequent()[0].replace('(?x)', subject, 1)
                            print(f"Tourist confirmed as {tourist_type}.")
                            return data
                        else:
                            print("Tourist type not confirmed. Let's try again.")
                            return None
    return None


def backward_chain_with_questions(rules, hypothesis, known_facts=set(), verbose=False):
    """
    Backward chaining with dynamic questions to confirm the hypothesis.
    
    Args:
        - rules: The set of rules used for inference.
        - hypothesis: The hypothesis we're trying to prove.
        - known_facts: A set of facts we already know.
        - verbose: If True, prints detailed steps.
    
    Returns:
        - A set of proven facts if the hypothesis can be proved.
        - "Hypothesis cannot be proved" if it cannot be proved.
    """
    subject = hypothesis.split()[0]  # Extract subject from the hypothesis (e.g., "Naruto")
    
    if verbose:
        print(f"Attempting to prove: {hypothesis}")
    
    # Check if the hypothesis is already known
    if hypothesis in known_facts:
        if verbose:
            print(f"Known fact: {hypothesis} is already proven.")
        return known_facts

    # Step 1: Find a rule where the hypothesis is the consequent
    for rule in rules:
        consequent = rule.consequent()[0].replace('(?x)', subject)

        if consequent == hypothesis:
            if verbose:
                print(f"Rule found that can prove the hypothesis: {rule}")
            antecedents = rule.antecedent()

            if isinstance(antecedents, str):
                antecedents = [antecedents]  # Ensure antecedents is a list

            # Step 2: Ask yes/no question for the first antecedent
            first_antecedent = antecedents[0].replace('(?x)', subject)
            if first_antecedent not in known_facts:
                yes_no_question = generate_yes_no_question(subject, first_antecedent)
                confirmation = input(yes_no_question).strip().lower()
                if confirmation == 'yes':
                    known_facts.add(first_antecedent)
                else:
                    print(f"Failed to prove antecedent: {first_antecedent}")
                    return "Hypothesis cannot be proved"

            # Step 3: Find a rule where the second antecedent is a consequent (starts with "is a")
            second_antecedent = antecedents[1].replace('(?x)', subject)
            for next_rule in rules:
                next_consequent = next_rule.consequent()[0].replace('(?x)', subject)
                if second_antecedent == next_consequent:
                    # Step 4: Ask specialized multiple-choice question for one correct antecedent
                    is_correct = generate_backward_multiple_choice_question(subject, next_rule.antecedent()[0], next_rule.antecedent())
                    if is_correct:
                        known_facts.add(next_rule.antecedent()[0].replace('(?x)', subject))
                        
                        # Step 5: Ask open-ended question for the remaining antecedent
                        remaining_antecedent = [ant.replace('(?x)', subject) for ant in next_rule.antecedent() if ant != next_rule.antecedent()[0]][0]
                        final_answer = input(f"Provide more details about {subject}: ").strip().lower()
                        if remaining_antecedent in final_answer:
                            known_facts.add(remaining_antecedent)
                            # Add the second antecedent to known facts
                            known_facts.add(second_antecedent)
                            # If both antecedents are proven, the hypothesis is true
                            known_facts.add(hypothesis)
                            print(f"{subject} is a {hypothesis.split()[2]} confirmed.")  # Confirmation message
                            return known_facts
                    else:
                        print(f"Failed to prove antecedent: {second_antecedent}")
                        return "Hypothesis cannot be proved"
    return "Hypothesis cannot be proved"


def instantiate(template, values_dict):
    """
    Given an expression ('template') with variables in it,
    replace those variables with values from values_dict.

    For example:
    >>> instantiate("sister (?x) {?y)", {'x': 'Lisa', 'y': 'Bart'})
    => "sister Lisa Bart"
    """
    if (isinstance(template, AND) or isinstance(template, OR) or
        isinstance(template, NOT)):

        return template.__class__(*[populate(x, values_dict) 
                                    for x in template])
    # elif isinstance(template, basestring):
    elif isinstance(template, str):
        return AIStringToPyTemplate(template) % values_dict
    else: raise ValueError ("Don't know how to populate a %s" % type(template))

# alternate name for instantiate
populate = instantiate

def match(template, AIStr):
    """
    Given two strings, 'template': a string containing variables
    of the form '(?x)', and 'AIStr': a string that 'template'
    matches, with certain variable substitutions.

    Returns a dictionary of the set of variables that would need
    to be substituted into template in order to make it equal to
    AIStr, or None if no such set exists.
    """
    try:
        return re.match( AIStringToRegex(template), 
                         AIStr ).groupdict()
    except AttributeError: # The re.match() expression probably
                           # just returned None
        return None

def is_variable(str):
    """Is 'str' a variable, of the form '(?x)'?"""
    # return isinstance(str, basestring) and str[0] == '(' and \
    #   str[-1] == ')' and re.search( AIStringToRegex(str) )
    return isinstance(str) and str[0] == '(' and \
      str[-1] == ')' and re.search( AIStringToRegex(str) )

def variables(exp):
    """
    Return a dictionary containing the names of all variables in
    'exp' as keys, or None if there are no such variables.
    """
    try:
        return re.search( AIStringToRegex(exp).groupdict() )
    except AttributeError: # The re.match() expression probably
                           # just returned None
        return None
        
class IF(object):
    """
    A conditional rule.

    This should have the form IF( antecedent, THEN(consequent) ),
    or IF( antecedent, THEN(consequent), DELETE(delete_clause) ).

    The antecedent is an expression or AND/OR tree with variables
    in it, determining under what conditions the rule can fire.

    The consequent is an expression or list of expressions that
    will be added when the rule fires. Variables can be filled in
    from the antecedent.

    The delete_clause is an expression or list of expressions
    that will be deleted when the rule fires. Again, variables
    can be filled in from the antecedent.
    """
    def __init__(self, conditional, action = None, 
                 delete_clause = ()):
        # Deal with an edge case imposed by type_encode()
        if type(conditional) == list and action == None:
            return self.__init__(*conditional)

        
        # Allow 'action' to be either a single string or an
        # iterable list of strings
        # if isinstance(action, basestring):
        if isinstance(action, str):
            action = [ action ]

        self._conditional = conditional
        self._action = action
        self._delete_clause = delete_clause

    def apply(self, rules, apply_only_one=False, verbose=False):
        """
        Return a new set of data updated by the conditions and
        actions of this IF statement.

        If 'apply_only_one' is True, after adding one datum,
        return immediately instead of continuing. This is the
        behavior described in class, but it is slower.
        """
        new_rules = set(rules)
        old_rules_count = len(new_rules)
        bindings = RuleExpression().test_term_matches(
            self._conditional, new_rules)

        for k in bindings:
            for a in self._action:
                new_rules.add( populate(a, k) )
                if len(new_rules) != old_rules_count:
                    if verbose:
                        print("Rule:", self)
                        print("Added:", populate(a, k))
                    if apply_only_one:
                        return tuple(sorted(new_rules))
            for d in self._delete_clause:
                try:
                    new_rules.remove( populate(d, k) )
                    if len(new_rules) != old_rules_count:
                        if verbose:
                            print("Rule:", self)
                            print("Deleted:", populate(d, k))
                        if apply_only_one:
                            return tuple(sorted(new_rules))
                except KeyError:
                    pass
                    
        return tuple(sorted(new_rules)) # Uniquify and sort the
                                        # output list


    def __str__(self):
        return "IF(%s, %s)" % (str(self._conditional), 
                               str(self._action))

    def antecedent(self):
        return self._conditional

    def consequent(self):
        return self._action

    __repr__ = __str__

class RuleExpression(list):
    """
    The parent class of AND, OR, and NOT expressions.

    Just like Sums and Products from lab 0, RuleExpressions act
    like lists wherever possible. For convenience, you can leave
    out the brackets when initializing them: AND([1, 2, 3]) ==
    AND(1, 2, 3).
    """
    def __init__(self, *args):
        if (len(args) == 1 and isinstance(args[0], list)
            and not isinstance(args[0], RuleExpression)):
            args = args[0]
        list.__init__(self, args)
    
    def conditions(self):
        """
        Return the conditions contained by this
        RuleExpression. This is the same as converting it to a
        list.
        """
        return list(self)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, 
                           ', '.join([repr(x) for x in self]) )

    __repr__ = __str__
        
    def test_term_matches(self, condition, rules, 
                          context_so_far = None):
        """
        Given an expression which might be just a string, check
        it against the rules.
        """
        rules = set(rules)
        if context_so_far == None: context_so_far = {}

        # Deal with nesting first If we're a nested term, we
        # already have a test function; use it
        # if not isinstance(condition, basestring):
        if not isinstance(condition, str):
            return condition.test_matches(rules, context_so_far)

        # Hm; no convenient test function here
        else:
            return self.basecase_bindings(condition, 
                                          rules, context_so_far)

    def basecase_bindings(self, condition, rules, context_so_far):
        for rule in rules:
            bindings = match(condition, rule)
            if bindings is None: continue
            try:
                context = NoClobberDict(context_so_far)
                context.update(bindings)
                yield context
            except ClobberedDictKey:
                pass

    def get_condition_vars(self):
        if hasattr(self, '_condition_vars'):
            return self._condition_vars

        condition_vars = set()

        for condition in self:
            if isinstance(condition, RuleExpression):
                condition_vars |= condition.get_condition_vars()
            else:
                condition_vars |= AIStringVars(condition)
                
        return condition_vars

    def test_matches(self, rules):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) == type(other) and list.__eq__(self, other)

    def __hash__(self):
        return hash((self.__class__.__name__, list(self)))

class AND(RuleExpression):
    """A conjunction of patterns, all of which must match."""
    class FailMatchException(Exception):
        pass
    
    def test_matches(self, rules, context_so_far = {}):
        return self._test_matches_iter(rules, list(self))

    def _test_matches_iter(self, rules, conditions = None, 
                           cumulative_dict = None):
        """
        Recursively generate all possible matches.
        """
        # Set default values for variables.  We can't set these
        # in the function header because values defined there are
        # class-local, and we need these to be reinitialized on
        # each function call.
        if cumulative_dict == None:
            cumulative_dict = NoClobberDict()

        # If we have no more conditions to analyze, pass the
        # dictionary that we've accumulated back up the
        # function-call stack.
        if len(conditions) == 0:
            yield cumulative_dict
            return
            
        # Recursive Case
        condition = conditions[0]
        for bindings in self.test_term_matches(condition, rules,
                                               cumulative_dict):
            bindings = NoClobberDict(bindings)
            
            try:
                bindings.update(cumulative_dict)
                for bindings2 in self._test_matches_iter(rules,
                  conditions[1:], bindings):
                    yield bindings2
            except ClobberedDictKey:
                pass

            
class OR(RuleExpression):
    """A disjunction of patterns, one of which must match."""
    def test_matches(self, rules, context_so_far = {}):
        for condition in self:
            for bindings in self.test_term_matches(condition, rules):
                yield bindings

class NOT(RuleExpression):
    """A RuleExpression for negation. A NOT clause must only have
    one part."""
    def test_matches(self, data, context_so_far = {}):
        assert len(self) == 1 # We're unary; we can only process
                              # one condition

        try:
            new_key = populate(self[0], context_so_far)
        except KeyError:
            new_key = self[0]

        matched = False
        for x in self.test_term_matches(new_key, data):
            matched = True

        if matched:
            return
        else:
            yield NoClobberDict()


class THEN(list):
    """
    A THEN expression is a container with no interesting semantics.
    """
    def __init__(self, *args):
        if (len(args) == 1 and isinstance(args[0], list)
            and not isinstance(args[0], RuleExpression)):
            args = args[0]
        super(list, self).__init__()
        for a in args:
            self.append(a)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join([repr(x) for x in self]) )

    __repr__ = __str__


class DELETE(THEN):
    """
    A DELETE expression is a container with no interesting
    semantics. That's why it's exactly the same as THEN.
    """
    pass

def uniq(lst):
    """
    this is like list(set(lst)) except that it gets around
    unhashability by stringifying everything.  If str(a) ==
    str(b) then this will get rid of one of them.
    """
    seen = {}
    result = []
    for item in lst:
        # if not seen.has_key(str(item)):
        if not str(item) in seen:
            result.append(item)
            seen[str(item)]=True
    return result

def simplify(node):
    """
    Given an AND/OR tree, reduce it to a canonical, simplified
    form, as described in the lab.

    You should do this to the expressions you produce by backward
    chaining.
    """
    if not isinstance(node, RuleExpression): return node
    branches = uniq([simplify(x) for x in node])
    if isinstance(node, AND):
        return _reduce_singletons(_simplify_and(branches))
    elif isinstance(node, OR):
        return _reduce_singletons(_simplify_or(branches))
    else: return node

def _reduce_singletons(node):
    if not isinstance(node, RuleExpression): return node
    if len(node) == 1: return node[0]
    return node

def _simplify_and(branches):
    for b in branches:
        if b == FAIL: return FAIL
    pieces = []
    for branch in branches:
        if isinstance(branch, AND): pieces.extend(branch)
        else: pieces.append(branch)
    return AND(*pieces)

def _simplify_or(branches):
    for b in branches:
        if b == PASS: return PASS
    pieces = []
    for branch in branches:
        if isinstance(branch, OR): pieces.extend(branch)
        else: pieces.append(branch)
    return OR(*pieces)

PASS = AND()
FAIL = OR()

