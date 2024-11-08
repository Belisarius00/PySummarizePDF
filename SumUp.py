import re
import os
import nltk
import pdfplumber
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#Depois da 1a vez podes apagar 
#nltk.download("punkt")
#nltk.download("averaged_perceptron_tagger")
#até aqui

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

CHUNK_SIZE = 512  # Tenho q ver o limite mas 512 funcemina bem

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split() #separa a frase (string grande) em palavras individuais (strings pequenos)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_pdf(pdf_path):
    text_data = []
    with pdfplumber.open(pdf_path) as pdf: #semelhante a with open(file) as f
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text_data.append((page_num, text))
    return text_data


def summarize_text(text, max_length=100, min_length=5):
    """Summarizes a given text."""
    try:
        return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] #chama o modelo propriamente dito para processar todas as frases do texto
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

def extract_keywords(text, document_collection, top_n=10):
    """Extracts top keywords from the text using TF-IDF, adjusting for small document sets."""
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english') #Inicializa o Vectorizer e indica para ignorar as stop-words da Ling. Inglesa
    tfidf_matrix = vectorizer.fit_transform(document_collection)  # Aprende o vocab e cria uma matriz TF-IDF
    tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]) #Passa para Lingua Natural os resultados da matrix anterior
    sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True) #ordena pela frequência
    return [word for word, score in sorted_keywords[:top_n]]

def extract_definitions(text):
    """Extracts potential definitions based on keyword tagging."""
    sentences = nltk.sent_tokenize(text) #divide o texto em frases
    definitions = {} #inicializa a lista de definições vazia
    for sentence in sentences:
        if re.search(r"\b(is|refers to|means|stands for)\b", sentence): #procura texto dentro de \b(...)\b na frase
            words = nltk.word_tokenize(sentence) #divide frases em palavras
            tags = nltk.pos_tag(words) #identifica o tipo de palavra
            for word, tag in tags:
                if tag == 'NNP' or tag == 'NN':  # Compila definições existentes no pdf
                    definitions[word] = sentence
    return definitions

def create_addendum(pdf_path):
    """Generates a summarized addendum for a PDF."""
    pdf_text = extract_text_from_pdf(pdf_path)
    addendum = {"summary": [], "keywords": [], "definitions": {}}

    non_blank_texts = [text for page_num, text in pdf_text if text.strip()]  # Filter out blank pages

    for page_num, text in pdf_text:
        if not text.strip():  # Skip páginas em branco
            continue
        
        # Definição de Argumentos
        summary_length = min(len(text.split()), 100)
        chunks = chunk_text(text)
        # Sumarizar e concatenar os chunks
        chunk_summaries = [summarize_text(chunk, max_length=summary_length) for chunk in chunks]
        page_summary = " ".join(chunk_summaries)
        addendum["summary"].append(f"Page {page_num}: {page_summary}")

        # Extrair keywords
        keywords = extract_keywords(text, non_blank_texts)
        addendum["keywords"].extend([(keyword, page_num) for keyword in keywords])

        # Extrair definições
        definitions = extract_definitions(text)
        addendum["definitions"].update(definitions)

    # Juntar keywords
    unique_keywords = list(set(addendum["keywords"]))
    addendum["keywords"] = sorted(unique_keywords, key=lambda x: x[1])
    print("Summarized page #", page_num)

    return addendum

#Correr as funções propriamente ditas
#FALTA IMPLEMENTAR INTERFACE DE CURSES OU GRÁFICA

pdf_path = os.listdir("files")
print(pdf_path)
for file in pdf_path:
    print(file, "\nfiles/"+file)
    addendum = create_addendum("files/"+file)
    # Printing the results
    print("Summary:", addendum["summary"])
    print("\nKeywords by Page:", addendum["keywords"])
    print("\nDefinitions:", addendum["definitions"])
    f = open(file+".txt", "x")
    write_to_text = f"Summary: {addendum['summary']}\nKeywords by Page: {addendum['keywords']}\nDefinitions: {addendum['definitions']}"  
    f.write(write_to_text)
    f.close()

