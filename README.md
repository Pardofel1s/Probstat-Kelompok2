# Implementation of Naïve Bayes - Group 2

| NRP | Name |
|:---:|:----:|
| 5025241015 | Farrel Aqilla Novianto |
| 5025241153 | Kamal Zaky Adinata |
| 5025241181 | Muhammad Naufal Hadaya Setiawan |
| 5025241212 | Akmal Yusuf |


## 1. Bayes' Theorem Overview

Bayes' Theorem is a fundamental mathematical formula used in probability and statistics. At its core, it provides a way to **update our beliefs or probabilities based on new evidence**.

Instead of just looking at the probability of an event happening in isolation, Bayes' Theorem helps us calculate the probability of an event happening given that another related event has already occurred.

### The Formula
The theorem is expressed mathematically with the following equation:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

To understand the formula, we need to break down what each piece represents. Let's assume $A$ is a hypothesis and $B$ is the observed evidence:
- $P(A|B)$ **(Posterior Probability)**: The probability of event $A$ being true, given that the evidence $B$ is present. This is usually the answer we are trying to find.
- $P(B|A)$ **(Likelihood)**: The probability of observing evidence $B$, given that hypothesis $A$ is true.
- $P(A)$ **(Prior Probability)**: Our initial belief in the probability of event $A$ occurring, before we saw any new evidence.
- $P(B)$ **(Marginal Probability)**: The total probability of observing the evidence $B$ under all possible scenarios (whether $A$ is true or not).

### A Classic Example: The Medical Test

The best way to grasp Bayes' Theorem is through the classic "rare disease" scenario.

Imagine a rare disease that affects **1%** of the population. A medical test for this disease is **99% accurate** (it correctly identifies 99% of sick people as positive, and 99% of healthy people as negative).

If a randomly selected person tests positive, what is the actual probability that they have the disease?

Human intuition often jumps straight to "99%." However, Bayes' Theorem reveals a very different reality. Let's plug in the numbers:
- $A$: The patient has the disease. Therefore, $P(A) = 0.01$ (Prior).
- $B$: The test result is positive.
- $P(B|A)$: The test is positive given the patient is sick = $0.99$ (Likelihood).
- $P(B)$: The total probability of getting a positive test. This happens if a sick person tests positive ($0.01 \cdot 0.99$) PLUS if a healthy person gets a false positive ($0.99 \cdot 0.01$). So, $P(B) = 0.0099 + 0.0099 = 0.0198$.

Now, we use the formula:

$$P(A|B) = \frac{0.99 \cdot 0.01}{0.0198} = 0.5$$

**The Result:** Even with a 99% accurate test, because the disease itself is so rare, a person with a positive result only has a **50%** chance of actually having the disease. The "prior" (the rarity of the disease) heavily anchors the final probability.

## 2. Application of Bayes' Theorem in the Naïve Bayes Algorithm

The Naïve Bayes algorithm is a highly popular machine learning technique used primarily for classification tasks, such as spam detection, sentiment analysis, and categorizing news articles. It is a direct, practical application of Bayes' Theorem, but with one massive, simplifying assumption.

### The "Naïve" Assumption: Conditional Independence

In the real world, data points (features) are often correlated. For example, in an email, the word "Free" and the word "Money" frequently appear together.

The Naïve Bayes algorithm ignores this reality. It makes the **"naïve" assumption that every single feature is completely independent of every other feature**, given the class label.

While this assumption is almost never entirely true in practice, it simplifies the mathematical calculations enormously and, surprisingly, still results in highly accurate and incredibly fast classifications.

### The Math: From One Event to Many Features

Let's look at how Bayes' Theorem adapts to handle a machine learning dataset. Suppose we have a data point with multiple features $X = (x_1, x_2, \dots, x_n)$ and we want to predict its class $y$ (e.g., Spam or Not Spam).

Standard Bayes' Theorem looks like this:

$$P(y | x_1, x_2, \dots, x_n) = \frac{P(x_1, x_2, \dots, x_n | y) \cdot P(y)}{P(x_1, x_2, \dots, x_n)}$$

Calculating the exact likelihood of that specific combination of features—$P(x_1, x_2, \dots, x_n | y)$—is computationally impossible for large datasets. This is where the naïve assumption comes to the rescue. Because we assume the features are independent, we can simply multiply their individual probabilities together:

$$P(x_1, x_2, \dots, x_n | y) = P(x_1|y) \cdot P(x_2|y) \cdot \dots \cdot P(x_n|y) = \prod_{i=1}^{n} P(x_i | y)$$

When we substitute this back into Bayes' Theorem, the formula becomes much easier for a computer to process:

$$P(y | x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i | y)}{P(x_1, \dots, x_n)}$$

### The Classification Rule

When a Naïve Bayes model is trying to classify a new piece of data, it doesn't actually need to calculate the exact probability. It just needs to figure out which class has the highest probability.

Since the denominator $P(x_1, \dots, x_n)$ is the exact same for every class you are comparing, the algorithm simply drops it. The model calculates the proportional probability for each possible class $y$ and chooses the maximum value:

$$\hat{y} = \arg\max_y \left( P(y) \prod_{i=1}^{n} P(x_i | y) \right)$$

### Example: Student Major Classification

Let's say we look at a student's short bio containing just two words: "C++" and "Data". We want to classify it as $y=\text{Informatics}$ or $y=\text{Industrial}$.

1. **Calculate the Priors** ($P(y)$): Based on the student population data on campus, let's say 20% are Informatics Engineering students ($0.20$) and 80% are Industrial Engineering students ($0.80$).

2. **Calculate the Likelihoods** ($P(x_i|y)$): We look at our training data to find how often these words appear in the bios for each major.
   - Probability of "C++" given they are Informatics: $P(\text{C++}|\text{Informatics}) = 0.70$
   - Probability of "Data" given they are Informatics: $P(\text{Data}|\text{Informatics}) = 0.50$
   - Probability of "C++" given they are Industrial: $P(\text{C++}|\text{Industrial}) = 0.05$
   - Probability of "Data" given they are Industrial: $P(\text{Data}|\text{Industrial}) = 0.40$

3. **Apply the Formula:**
   - **Informatics Score:** $0.20 \cdot 0.70 \cdot 0.50 = 0.07$
   - **Industrial Score:** $0.80 \cdot 0.05 \cdot 0.40 = 0.016$

Because $0.07$ is significantly larger than $0.016$, the algorithm confidently classifies the student as an **Informatics Engineering** student.

## 3. Dataset Description

## 4. Naïve Bayes Implementation Results

## 5. Conclusion