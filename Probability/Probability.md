## Probability Bases
    * General Definition of Probability
        In simple terms, probability is a measure of how likely an event is to occur, ranging from 00 (impossible) to 11 (certain).
        For example:

            Probability of getting heads when flipping a fair coin = 0.50.5.

            Probability of rolling a 7 on a standard six-sided die = 00 (impossible).

        Your Phrase:

            "The probability of an event A given the event B has occurred"

        That is conditional probability:
        P(A∣B)=P(A∩B)P(B)
        P(A∣B)=P(B)P(A∩B)​

        It tells us the probability of AA happening under the condition that BB has already happened.

     * Bayes Theorem  
        Bayes Theorem is a formula that tells you how much you should trust a new piece of evidence based on your past experience.

        It answers the question: "Knowing what has happened in the past, how likely is it that this new thing is true?"
        The Simple English Breakdown

        Imagine you hear an alarm. You want to know: Is it a real fire, or just a false alarm?

        Bayes Theorem helps you calculate this by combining two pieces of information:

            The General Rule (The Past): How often does this area actually have fires? (Usually, very rarely).

            The Evidence (The Present): How reliable is this alarm? Does it go off every time there is a fire, or does it go off randomly?

        Bayes Theorem says:
        Even though the alarm is ringing, if fires are super rare and the alarm is a little bit faulty, the chance it's a real fire might still be pretty low.
        The Famous Formula

        You will usually see it written like this:
        P(A∣B)=P(B∣A)×P(A)P(B)
        P(A∣B)=P(B)P(B∣A)×P(A)​

        But let's translate those symbols into English:

            P(A|B): Your question. "What is the probability of Event A (Fire) given that Event B (Alarm) happened?"

            P(B|A): The reliability of the evidence. "What is the probability the Alarm goes off if there IS a Fire?" (This should be high, like 99%).

            P(A): Your past experience (The "Prior"). "What is the general probability of a Fire happening at all?" (This is usually very low, like 0.1%).

            P(B): The total chances of the evidence. "What is the total probability the Alarm goes off for any reason?" (This includes real fires AND false alarms).

    * Common Probabilty Distribution
        * Gaussian(Normal) Distribution
            The Two Key Numbers
                Every Normal Distribution is described by just two numbers:
                Term	Symbol	Plain English	Example (Heights)
                Mean	μ (mu)	The average - the center of the curve	165 cm
                Standard Deviation	σ (sigma)	The spread - how wide the curve is	10 cm

    * Bernoulli Distribution
        * Describes outcomes of a binary experiment
            Imagine flipping a coin. That's exactly what Bernoulli Distribution is all about!

            * The Simple Idea

                The Bernoulli Distribution is the simplest probability distribution possible - it only has two possible outcomes:
                Outcome	Coin Flip	Medical Test	Website
                Success (1)	Heads	Has disease	Clicks ad
                Failure (0)	Tails	No disease	Doesn't click
                Just One Number Controls Everything

                The entire distribution is controlled by one single number: p (probability of success) 

                p = probability of success (Heads)
                1-p = probability of failure (Tails)  
                    * Example: Fair coin

                        p = 0.5 (50% chance of Heads)

                        1-p = 0.5 (50% chance of Tails)

                    Example: Biased coin that loves Heads

                        p = 0.8 (80% chance of Heads)

                        1-p = 0.2 (20% chance of Tails) 

    * Binomial Distribution
        * Models the number of success in N independent of Bernoulli trials  
        * Bernoulli vs Binomial: The Coffee Shop Analogy
                Bernoulli	Binomial
            Question	"Will the next customer order coffee?"	"How many of the next 10 customers will order coffee?"
            Scope	Just one customer	A group of customers
            Outcome	Yes (1) or No (0)	A number from 0 to 10   

        * The Three Key Ingredients

    * Poisson Distribution
        * model the number of events in a fixed interval of time or space 
        * The Simple Idea
            The Poisson Distribution (pronounced "Pwah-sohn") answers the question: "How many times will something happen in a fixed amount of time or space?"

            Unlike Binomial (which counts successes in a fixed number of trials), Poisson counts events in a continuous interval.

            * Real-Life Examples First
                Scenario	What's Being Counted	The Interval
                ☕ Coffee shop	Number of customers arriving	Per hour
                🚑 Hospital	Number of emergency calls	Per day
                🌧️ Weather	Number of rainy days	Per month
                📧 Email	Number of spam emails	Per day
                ☎️ Call center	Number of phone calls	Per minute
                ⭐ Galaxy	Number of stars	Per square light-year

            * The One Number That Controls Everything
                Just like Bernoulli had p, Poisson has λ (lambda - the Greek letter L):
                Symbol	Name	Plain English
                λ	Lambda	Average rate of events in that interval

                Example: If a coffee shop gets 20 customers per hour on average:

                    λ = 20 customers/hour

                    We can ask: "What's the probability of getting exactly 15 customers in the next hour?"

            * The Key Insight
                Poisson Distribution assumes events happen:

                    Independently (one customer doesn't affect another)

                    Randomly (no patterns)

                    At a constant average rate (λ stays the same)

                Think of it like raindrops on a window - they fall randomly, independently, but at a steady average rate.

    * Measures of Central Tendency and Dispersion
        * Central Tendecy

            * Mean:
                The Mean is just a fancy word for what most people call the "Average."
                Imagine you have 5 friends, and they have this much money:

                    Friend A: $5

                    Friend B: $10

                    Friend C: $2

                    Friend D: $5

                    Friend E: $3

                To find the "Mean," you pretend you dump all the money in a pile and then split it equally among everyone.

                    Add it all up: $5 + $10 + $2 + $5 + $3 = $25 total.

                    Count the people: There are 5 friends.

                    Divide: $25 ÷ 5 = $5.

                The Mean is $5. It means if everyone shared perfectly, each person would have 5 bucks.

            * Median:
                The Median is the "Middle Number." It's the person standing right in the middle of the line.
                Using the same money from above, you have to line your friends up from the one with the least money to the one with the most money.

                    Line them up (Sort the numbers):
                    $2 (Least) → $3 → $5 → $5 → $10 (Most)

                    Find the Middle: Look at the line. The person right in the middle (the 3rd person in line) has $5.

                The Median is $5.

                Why is this cool? If Bill Gates suddenly became your friend and joined the line with $1,000, the "Mean" would go crazy high. But the "Median" (the middle kid) would still be standing around the same spot. It shows you what the "typical" person has without the super-rich or super-poor messing up the view.

            * Mode: 
                The Mode is the "Most Popular" number. It’s the value that shows up the most times.
                Look at our friends' money again:
                $5, $10, $2, $5, $3.

                Which number appears more than the others?

                    $5 appears twice.

                    All the other numbers appear only once.

                The Mode is $5.

                Real-life example: If you asked the class what their favorite ice cream is, and 10 people said "Chocolate," 3 said "Vanilla," and 2 said "Strawberry," the Mode would be Chocolate because it's the most popular choice. 

    * Hypothesis Testing
        What is hypothesis?
            A Hypothesis is just a fancy word for an "Educated Guess."
            It's what you believe is true about the world before you check the data.

        * Formulate the Hypothesis

            * Null hypothesis:
                This is the idea that "Nothing is going on." Everything is normal. There is no change.
                Coin example: "The coin is perfectly normal. It lands on Heads 50% of the time."
                School example: "Boys and girls have the same average height in this class."

            * Althernative Hypothesis:
                This is the idea that "Something IS going on." There is a change, a difference, or an effect.
                Coin example: "The coin IS rigged! It does NOT land on Heads 50% of the time."
                School example: "Boys are taller than girls in this class."

        * Claculate Test Statistic
            Now you have to gather evidence. You flip the coin 100 times.
            You count the flips.
            You do some math (the "Test Statistic") that turns your flips into a single number that represents "how weird" the results are.
        Let's say you flipped it 100 times and got 60 Heads and 40 Tails.
            Is that weird? A little bit.
            Is it weird enough to prove the coin is rigged? Maybe. That's what the next step figures out. 

        * Determine P-value
            The P-value is the most important part. It answers the question:
        "If the Null Hypothesis is true (if the coin is actually fair), what are the chances I got a result this weird just by random luck?"
            P-value = 0.01 (Very Low): "There is only a 1% chance I got 60 Heads if the coin is fair. Wow! That means the coin is probably NOT fair."
            P-value = 0.30 (High): "There is a 30% chance I got 60 Heads if the coin is fair. You know what? That's pretty easy to do by accident. The coin is probably fine."

        * Interpret Results 
            Finally, you compare the P-value to a rule. Scientists usually use a rule called "0.05" (or 5%).

        If P-value < 0.05 (Small):

            Meaning: "The chances of this being random are super tiny."

            Your Decision: "Reject the Null." You kick the boring guess out. You say, "The coin IS rigged!" (You accept the Alternative).

        If P-value > 0.05 (Big):

            Meaning: "Eh, this could easily happen by random chance."

            Your Decision: "Fail to reject the Null." You say, "We don't have enough proof. The coin is probably fair." (You stick with the boring guess).  

    * Confidence Intervals and Statistical Significancy
        * Confidence intervals
            * Range of values within which the true population parameters is expected to lie                              

                           