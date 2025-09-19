import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)
    sentence_scores = {}
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_scores[sentence] = sum(word_freq[word] for word in sentence_lower.split() if word in word_freq)
    summarized_sentences  = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summarized_sentences)

    return summary
text = """
Technology has revolutionized the field of education in remarkable ways. Today, classrooms are no longer confined to four walls, 
as digital platforms make learning accessible anywhere. Online classes and e-learning tools connect teachers and students beyond 
geographical limits. With the help of technology, concepts are explained using animations, videos, and simulations that make learning 
more interactive. Students can access endless information through the internet within seconds. Tools like smartboards, projectors, and
tablets enhance the teaching process. Technology also supports personalized learning, where each student can progress at their own pace. 
Learning management systems help teachers track student performance efficiently. Online quizzes and assignments save time and provide 
instant feedback. Virtual reality and augmented reality are being introduced to give real-life learning experiences. Collaboration has 
also become easier through digital group projects and discussion forums. Technology bridges the gap between students of different abilities 
by offering special learning apps. Mobile apps make revision and practice possible anytime, anywhere. Educational YouTube channels and online 
courses provide free resources for self-learners. Teachers also benefit as they can share resources widely and improve their teaching techniques. 
Technology promotes creativity, as students can design presentations, projects, and even code programs. It helps prepare learners for future careers
 where digital skills are essential. At the same time, it develops problem-solving and critical-thinking abilities. Overall, technology has transformed
  education into a more flexible, engaging, and student-centered process.
"""
summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base', framework='pt')
if len(text.split()) > 500:
    text = " ".join(text.split()[:500])
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print("\nAbstractive Summary:")
print(summary[0]['summary_text'])