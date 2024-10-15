import random
import string
import spacy

# Load the English model for spacy
nlp = spacy.load("en_core_web_sm")

class Perturbation:

    """Base class for random perturbations."""

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable

    def change_verb_tense(self, s):
        """Change verb tenses in the given sentence."""
        
        # Process the text using spacy
        doc = nlp(s)
        modified_tokens = []

        # Loop over each token and change the tense if it's a verb
        for token in doc:
            if token.pos_ == "VERB":
                # Simple transformation: if the verb is in present tense, change to past tense, and vice versa
                if token.tag_ in ["VB", "VBP", "VBZ"]:  # Present tense
                    modified_tokens.append(token.lemma_ + "ed")  # Change to past tense
                elif token.tag_ in ["VBD", "VBN"]:  # Past tense
                    modified_tokens.append(token.lemma_)  # Change to present tense
                else:
                    modified_tokens.append(token.text)
            else:
                modified_tokens.append(token.text)

        # Join the modified tokens to form the modified sentence
        modified_sentence = " ".join(modified_tokens)
        
        return modified_sentence


class RandomSwapPerturbation(Perturbation):

    """Implementation of random swap perturbations with verb tense changes.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        # Apply Random Swap Perturbation
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        perturbed_string = ''.join(list_s)
        
        # Apply Verb Tense Changes on the perturbed result
        tense_changed_string = self.change_verb_tense(perturbed_string)
        
        return tense_changed_string


class RandomPatchPerturbation(Perturbation):

    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        return ''.join(list_s)


class RandomInsertPerturbation(Perturbation):

    """Implementation of random insert perturbations.
    See `RandomInsertPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return ''.join(list_s)


