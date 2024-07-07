from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


class PDFReader():
    """
    PDFReader is a utility class designed to load and read PDF documents from a specified directory.

    This class leverages the PyPDFDirectoryLoader to load PDF files and extract their content.
    It provides a simple interface to initialize with a directory path and load the documents,
    returning the content of each page in the documents.

    Attributes:
        directory (str): The path to the directory containing PDF files to be loaded.

    Methods:
        __init__(directory: str):
            Initializes the PDFReader with the specified directory path.

        load() -> List[str]:
            Loads the PDF documents from the directory and returns a list of page contents.
    """

    def __init__(self, directory: str):
        self.directory = directory

    def load(self):
        loader = PyPDFDirectoryLoader(self.directory)
        documents = loader.load()
        return [doc.page_content for doc in documents]
