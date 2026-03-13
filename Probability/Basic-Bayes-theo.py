def bayes_theo(prior, likelihood, evidence):
    return(likelihood * prior) / evidence

# Your phone won't turn on. Is it broken?
prior = 0.01      # Only 1% of phones are actually broken
likelihood = 0.99 # If broken, it almost always won't turn on
evidence = 0.10   # 10% of the time phones won't turn on (dead battery, broken, etc.)

result = bayes_theo(prior, likelihood, evidence)
print(f"Chance phone is actually broken: {result:.1%}")