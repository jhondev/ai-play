from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "I love this course, it is amazing!"

result = classifier(text)[0]

print(f"The text \"{text}\" was classified as {result['label']} with a score of {round(result['score'], 4) * 100}%")