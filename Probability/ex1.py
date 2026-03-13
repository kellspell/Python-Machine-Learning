def bayes_theo(prior, sensitivity, specificity):
    evidence = (sensitivity * prior) + ((1 - specificity) * (1 - prior))
    posterior = (sensitivity * prior) / evidence
    return posterior

# Your phone won't turn on. Is it broken?
prior = 0.01      # Only 1% of phones are actually broken
sensitivity = 0.99 # If broken, it almost always won't turn on
specificity = 0.90   # 10% of the time phones won't turn on (dead battery, broken, etc.)

result = bayes_theo(prior, sensitivity, specificity)
print(f"Probability of desease is: {result:.1%}")