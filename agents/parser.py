from typing import Literal
import fitz as fz
from pathlib import Path
import base64
import logging

# Configure module-level logger (adjust level as needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# cSpell:ignore xobjects xobject  # technical PDF term used in PyMuPDF

# import custom exceptions
from custom_exceptions import IncompatibleFileFormatException, EmptyFileExtensionException, ImageEncodingException

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

def get_file_extension(file_path: Path) -> str:
    """Extracts file suffix (extension) and returns it as a string. Raises EmptyFileExtensionException if no extension found

    Args:
        file_path (Path): A Path object to the file

    Returns:
        str: Returns the extension (e.g. .png) as a string
    """

    extension = file_path.suffix.lower()
    # [CHANGED] Replaced print with logger for better observability
    logger.info(f"Extension of {file_path} is {extension}")

    if extension == '' or not extension:
        raise EmptyFileExtensionException(message=f"{file_path} does not have an extension; cannot proceed with further validation\n")
    
    return extension


def normalize_file_path(file_path: str) -> tuple[Path, str]:
    """Normalizes the provided path after validation and returns a Path object for further I/O operations

    Args:
        file_path (str): The file path to the document that needs to be processed

    Returns:
        tuple[Path, str]: A Path object of the file location and the extension of the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find the path specified: {file_path}")
    
    if not file_path.is_file():
        raise IsADirectoryError(f"Expected a file, but found a directory: {file_path}")
    

    try:
        extension = get_file_extension(file_path=file_path)
        return (file_path, extension)
    except Exception as e:
        # Re-raise to let caller handle; preserves context for debugging
        raise IncompatibleFileFormatException(
            message=f"Failed to normalize file path {file_path}: {e}"
        ) from e

def classify_file_type(extension: str) -> Literal["raster", "document"]:
    """Classifies the file for further processing and extraction
    """
    if extension in raster_formats:
        return "raster"
    elif extension in file_formats:
        return "document"
    else:
        raise IncompatibleFileFormatException(message=f"{extension} is not a supported file type\n")

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

    normalized_file_path = normalize_file_path(file_path)
    file_path = normalized_file_path[0]
    extension = normalized_file_path[1]

    # [CHANGED] Use logger instead of print
    # logger.info(f"Extension of {file_path} is {extension}")

    file_type = classify_file_type(extension)

    # [CHANGED] Return as tuple for consistency with docstring
    return (file_path, True, file_type)


def extraction_branching(validated_file_tuple: tuple):
    """Directs the control flow to the appropriate functions for extraction

    Args:
        validated_file_tuple (tuple): The output from document_validator().
    """

    # explicitly check if the file has been validated
    if not validated_file_tuple[1]:
        print(f"Skipping processing for invalid file: {validated_file_tuple[0]}")  # Log intentional skip
        return
    
    if validated_file_tuple[1]:
        # the document is valid and can be further processed for extraction
        file_path = validated_file_tuple[0]
        file_type = validated_file_tuple[2]

        if file_type == "raster":
            # images need to be processed before sending into the model for extraction

            # encode the image and retrieve the base64 encoding
            try:
                base64_image_encoding = image_encoder(file_path)
            except Exception as image_encoding_func_call_exp:
                print(f"Exception occurred while calling the image_encoder() inside extraction_branching(): {image_encoding_func_call_exp.with_traceback()}\n")
                # after this, you need to send it over to the vision-first function

        elif file_type == "document":
            # documents need to be further processed for extraction
            print(f"Document: {file_path}")
            document = fz.open(filename=Path(file_path))

            """How the document will be flagged as scanned or not:
                - get the total count of page fonts
                - get the total count of XObjects in the page
                - check the number of pages where images take up more area
                - check the number of drawings and tables

                conditional logic for classification:
                    -> if the number of fonts > XObjects:
                        if the number of pages where images take up more area is lesser than half:
                            * page gets flagged as text and can be used for text extraction
                    -> else:
                        if the number of pages where images take up more area is more than half the count of pages:
                            if the number of drawings and images > 0:
                                * page is flagged as scanned and needs to go for visual extraction
                
                We are performing page level classification and not on the entire document
            """

            # track the pages flagged for text extraction
            text_extraction_flagged_page_count = []

            for page in document:
                # store the fonts, XObjects, and the text from the page
                page_fonts = set()
                page_xobjects = set()
                page_character_count = 0
                page_drawings, total_page_images = 0, 0

                page_character_count = len(page.get_text()) # get the number of characters in the page

                # retrieve the fonts and XObjects and add them to the respective sets
                for font in page.get_fonts():
                    page_fonts.add(font[0])
                for xobject in page.get_xobjects():
                    page_xobjects.add(xobject[0])
                

                # get the number of images and drawings
                page_images = page.get_images()
                total_page_images += len(page_images)
                page_drawings += len(page.get_drawings())

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
                
                if page_character_count > 150 and len(page_fonts) > 0:
                    text_extraction_flagged_page_count.append(True)
                elif page_character_count < 50 or is_page_image_dominant or len(page_xobjects) >= len(page_fonts):
                    text_extraction_flagged_page_count.append(False)
                else:
                    text_extraction_flagged_page_count.append(False)
                

            print(f"Number of pages flagged for text extraction: {len([page for page in text_extraction_flagged_page_count if page == True])}")
            print(f"Number of pages flagged for vision extraction: {len([page for page in text_extraction_flagged_page_count if page == False])}")


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
                raise ImageEncodingException(
                    message=f"Exception occurred while trying to encode the image: {err}"
                ) from err
    else:
        raise IncompatibleFileFormatException(f"{file_path_to_image} not found") # replace this with new error type

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
            print("-------\n")

if __name__ == "__main__":
    main()
