from modulefinder import test
from ntpath import isfile
import fitz as fz
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

# when a user upload an image file, it most probably will be in one of these formats
raster_formats = {
    ".jpg", ".jpeg", ".jpe", ".jif", ".jfif", 
    ".png", 
    ".webp", 
    ".tif", ".tiff", 
    ".heic", ".heif", 
    ".bmp", 
    ".raw", ".cr2", ".nef", ".arw"
}

# for any kind of documents that can be uploaded. for now only these formats are supported; later more can be accommodated
file_formats = {
    ".pdf", ".txt"
}

def document_validator(file_path: str) -> tuple[str, bool, str]:
    """Reads a document and returns whether it is a document or an image

    Args:
        file_path (str): The path to the document or image file. In the context of the agent, this will be the path to the input that the user has uploaded. This function is expected to account for all the different file types that the user might upload. 
        
        Given that there can be multiple types of documents that can be uploaded, and a number of file uploads, only the following formats are expected to be uploaded:

        Raster formats:
        - "jpg", "jpeg", "jpe", "jif", "jfif", 
        - "png", 
        - "webp",  
        - "tif", "tiff", 
        - "heic", "heif", 
        - "bmp", 
        - "raw", "cr2", "nef", "arw"

        File formats:
        - "pdf", "txt"

    Returns: tuple[str, bool, str]
        str: The file path. Returns None if the file is incompatible
        bool: Whether the file is a compatible format for the agent to process
        str: The type of file that was uploaded (raster, document). Returns None when incompatible file format uploaded 
    """

    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    print(f"Extension of {file_path} is {extension}")

    if extension in raster_formats:
        print(f"File {file_path} is a raster format\n")
        return (file_path, True, "raster")
    elif extension in file_formats:
        print(f"File {file_path} is a file format, extension: {extension}\n")
        return (file_path, True, "document")
    else:
        print(f"File {file_path} is not a compatible format\n")
        return (None, False, None)


def extraction_branching(validated_file_tuple: tuple):
    """Directs the control flow to the appropriate functions for extraction

    Args:
        validated_file_tuple (tuple): The output from document_validator().
    """
    
    if validated_file_tuple[1]:
        # the document is valid and can be further processed for extraction
        file_path = validated_file_tuple[0]

        if validated_file_tuple[2] == "raster":
            # images need to be processed before sending into the model for extraction

            # encode the image and retrieve the base64 encoding
            try:
                base64_image_encoding = image_encoder(file_path)
            except Exception as image_encoding_func_call_exp:
                print(f"Exception occurred while calling the image_encoder() inside extraction_branching(): {image_encoding_func_call_exp.with_traceback()}\n")
                # after this, you need to send it over to the vision-first function

        elif validated_file_tuple[2] == "document":
            # documents need to be further processed for extraction
            print(f"Document: {file_path}")
            document = fz.open(filename=Path(file_path))

            """How the document will be flagged as scanned or not:
                - get the total count of document fonts
                - get the total count of XObjects in the document
                - check the number of pages where images take up more area
                - check the number of drawings and tables

                conditional logic for classification:
                    -> if the number of fonts > XObjects:
                        if the number of pages where images take up more area is lesser than half:
                            * document gets flagged as text and can be used for text extraction
                    -> else:
                        if the number of pages where images take up more area is more than half the count of pages:
                            if the number of drawings and images > 0:
                                * document is flagged as scanned and needs to go for visual extraction
            """

            # store the fonts, XObjects, and the text from the document
            document_fonts = set()
            document_xobjects = set()
            document_text = ""
            document_drawings, document_images = 0, 0

            # track the pages where images take up more area
            image_dominant_pages = 0

            for page in document:
                document_text += " " + page.get_text()

                # retrieve the fonts and XObjects and add them to the respective sets
                for font in page.get_fonts():
                    document_fonts.add(font[0])
                for xobject in page.get_xobjects():
                    document_xobjects.add(xobject[0])
                

                # get the number of images and drawings
                page_images = page.get_images()
                document_images += len(page_images)
                document_drawings += len(page.get_drawings())

                # check the area occupied by images in a page compared to text
                page_area = abs(page.rect)
                is_page_image_dominant = False

                for image_tuple in page_images:
                    try:
                        image_rectangle = page.get_image_bbox(image_tuple)
                        if page_area > 0 and (abs(image_rectangle) / page_area) > 0.95:
                            is_page_image_dominant = True
                    except Exception:
                        continue
                
                if is_page_image_dominant:
                    image_dominant_pages += 1
                    
                
            print(f"Total number of fonts found: {len(document_fonts)}")
            print(f"Total number of XObjects found: {len(document_xobjects)}")
            print(f"Total number of drawings: {document_drawings}")
            print(f"Total number of images: {document_images}")
            print(f"Number of image dominant pages: {image_dominant_pages} / {document.page_count}\n")


    elif validated_file_tuple[1] == False:
        print(f"Incompatible file uploaded.\n")


def image_encoder(file_path_to_image: str) -> str:
    """Encode an uploaded image or scanned document page to base64 string

    Args:
        file_path_to_image (str): The path to the image that needs to be encoded

    Returns:
        str: The resultant base64 string
    """

    path = Path(file_path_to_image)
    if path.exists():
        # read the image
        with open(path, "rb") as image:
            try:
                image_b64_encodedString =  base64.b64encode(image.read()).decode()
                return image_b64_encodedString
            except Exception as err:
                print(f"Exception occurred while trying to encode the image: {err.with_traceback()}\n")
                return None
    else:
        print(f"{file_path_to_image} is invalid; not found\n")
        return None

def main():
    """The main function of the Document Parser agent
    """
    print("Document parser agent execution commenced...\n")

    # contains the testing documents
    testing_data_directory = Path("./test_data")
    
    for file_path in testing_data_directory.iterdir():
        if file_path.is_file():

            # start by performing the document validation
            validation_tuple = document_validator(file_path)

            # pass the validation tuple to the extractor branching function
            extraction_branching(validation_tuple)

if __name__ == "__main__":
    main()