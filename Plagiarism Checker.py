import textract
import fitz  # PyMuPDF for PDF handling
from PyQt5 import QtWidgets, QtGui
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import spacy

nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from Word document using textract
def extract_text_from_word(docx_path):
    text = textract.process(docx_path).decode('utf-8')  # Textract handles .docx extraction
    return text

# Function to fetch content from a website
def fetch_website_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    return ""

# Function to scrape Google search results directly
def google_search(query):
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Mimic a real browser request
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        
        # Extract search result URLs from Google (limit to a few results)
        for result in soup.find_all('a', href=True):
            link = result['href']
            if 'http' in link:
                search_results.append(link.split('&')[0].replace('/url?q=', ''))
        
        return search_results[:5]  # Return top 5 results
    else:
        return []

# Function to calculate similarity using TF-IDF
def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(vectors)[0][1]

# PyQt GUI class
class PlagiarismDetectorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Plagiarism Detection Tool")
        self.setGeometry(100, 100, 800, 600)
        

        # Add widgets
        self.text_area_1 = QtWidgets.QTextEdit(self)
        self.text_area_1.setGeometry(50, 50, 700, 150)

        self.compare_button = QtWidgets.QPushButton("Check for Plagiarism", self)
        self.compare_button.setGeometry(50, 450, 200, 30)
        self.compare_button.clicked.connect(self.compare_with_websites)

        self.upload_file_button = QtWidgets.QPushButton("Upload File", self)
        self.upload_file_button.setGeometry(300, 450, 150, 30)
        self.upload_file_button.clicked.connect(self.upload_file)

        self.result_label = QtWidgets.QLabel("Similarity: N/A", self)
        self.result_label.setGeometry(50, 500, 700, 30)

    def upload_file(self):
        # Use QFileDialog to open a file dialog for the user to select a file
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf);;Word Files (*.docx)")
        
        if file_name:
            if file_name.endswith(".pdf"):
                text = extract_text_from_pdf(file_name)
            elif file_name.endswith(".docx"):
                text = extract_text_from_word(file_name)
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported File", "Please upload a PDF or Word document.")
                return

            self.text_area_1.setText(text)

    def compare_with_websites(self):
        doc1 = self.text_area_1.toPlainText().strip()

        if not doc1:
            QtWidgets.QMessageBox.warning(self, "Error", "Document must be provided!")
            return

        # Search for matching websites using Google search (first 200 characters)
        similar_websites = google_search(doc1[:200])
        
        # Check if we found any similar websites
        if similar_websites:
            for website in similar_websites:
                web_content = fetch_website_content(website)
                if web_content:
                    similarity_score = calculate_similarity(doc1, web_content)
                    if similarity_score > 0.2:  # Arbitrary threshold for similarity
                        highlighted_text = self.highlight_matching_text(doc1, web_content)
                        self.text_area_1.setHtml(highlighted_text)
                        self.result_label.setText(f"Similarity with {website}: {similarity_score:.2f}")
                        return
            QtWidgets.QMessageBox.information(self, "No Match Found", "No significant match found on the web.")
        else:
            QtWidgets.QMessageBox.information(self, "No Websites Found", "No websites returned from the search.")

    def highlight_matching_text(self, doc1, web_content):
        """Highlight similar sections between the document and the fetched web content"""
        tokens1 = set(doc1.split())
        tokens2 = set(web_content.split())
        common_tokens = tokens1.intersection(tokens2)

        highlighted = doc1
        for token in common_tokens:
            highlighted = highlighted.replace(token, f"<span style='background-color: yellow'>{token}</span>")

        return highlighted

# Run the application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = PlagiarismDetectorApp()
    window.show()
    sys.exit(app.exec_())
