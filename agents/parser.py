from dataclasses import dataclass
from typing import Literal
import fitz as fz
from pathlib import Path
import base64
import logging
import io
from PIL import Image

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

# define data classes to store the contents of the processed document
@dataclass
class ProcessedPage:
    page_number: int
    extraction_method: Literal["text", "vision"]
    raw_content: str | None # populated by the text extraction and stored as a string
    image_b64_encoded: str | None # populated by the vision extraction which is b64 encoded

@dataclass # similar data class but for raster files
class ProcessedImage:
    file_path: str
    extraction_method: Literal["vision"]
    image_b64_encoded: str | None
@dataclass
class ProcessedDocument:
    file_path: str
    total_pages: int
    pages: list[ProcessedPage]

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

# a helper function to render a page as a b64 encoded byte string
def render_page_to_b64(page: fz.page) -> str:
    """Renders a provided page as a b64 encoded string of bytes

    Args:
        page (fz.page): The page that needs to be encoded as an image

    Returns:
        str: b64 encoded byte string
    """
    page_matrix = fz.Matrix(2, 2)
    page_pix = page.get_pixmap(matrix=page_matrix)
    return image_encoder(pix.tobytes("png"))

def extraction_branching(validated_file_tuple: tuple):
    """Directs the control flow to the appropriate functions for extraction

    Args:
        validated_file_tuple (tuple): The output from document_validator().
    """

    # explicitly check if the file has been validated
    if not validated_file_tuple[1]:
        logger.critical(f"Skipping processing for invalid file: {validated_file_tuple[0]}")
        # print(f"Skipping processing for invalid file: {validated_file_tuple[0]}")  # Log intentional skip
        return
    
    if validated_file_tuple[1]:
        # the document is valid and can be further processed for extraction
        file_path = validated_file_tuple[0]
        file_type = validated_file_tuple[2]

        if file_type == "raster":
            # images need to be processed before sending into the model for extraction

            image_byte_arr = io.BytesIO()

            # encode the image and retrieve the base64 encoding
            try:
                image = Image.open(file_path)
                # save the image bytes to the byte array
                image.save(image_byte_arr, format='PNG')
                retrieved_image_bytes = image_byte_arr.getvalue()

                base64_image_encoding = image_encoder(retrieved_image_bytes)

                processed_raster_file = ProcessedImage(
                    file_path = file_path,
                    extraction_method = "vision",
                    image_b64_encoded = base64_image_encoding
                )
                
                return processed_raster_file

            except Exception as image_encoding_func_call_exp:
                # print(f"Exception occurred while calling the image_encoder() inside extraction_branching(): {image_encoding_func_call_exp.with_traceback()}\n")
                logger.error(f"Exception in image_encoder(): {image_encoding_func_call_exp}", exc_info=True)
                # after this, you need to send it over to the vision-first function

        elif file_type == "document":
            # documents need to be further processed for extraction
            # print(f"Document: {file_path}")
            logger.info(f"Document: {file_path}")
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
            processed_page_data_list: list[ProcessedPage] = []

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

                    page_text = page.get_text()

                    processed_page_data = ProcessedPage(
                        page_number = page.number,
                        extraction_method = "text",
                        raw_content = page_text,
                        image_b64_encoded = None
                    )
                    processed_page_data_list.append(processed_page_data)

                elif page_character_count < 50 or is_page_image_dominant or len(page_xobjects) >= len(page_fonts):
                    text_extraction_flagged_page_count.append(False)

                    # # convert the current page into an image and pass it into the b64 image encoder function
                    # page_matrix = fz.Matrix(2, 2)
                    # page_pix = page.get_pixmap(matrix=page_matrix)
                    
                    # # retrieve the image bytes and pass it into the function
                    # page_image_bytes = page_pix.tobytes("png")
                    b64_encoded_page_image = render_page_to_b64(page)

                    processed_page_image_data = ProcessedPage(
                        page_number = page.number,
                        extraction_method = "vision",
                        raw_content = None,
                        image_b64_encoded = b64_encoded_page_image
                    )
                    processed_page_data_list.append(processed_page_image_data)

                else:
                    logger.warning(f"Page {page.number} is ambiguous (char_count={page_character_count}, font_count={len(page_fonts)}) - routing to vision")
                    text_extraction_flagged_page_count.append(False)
                    # # convert the current page into an image and pass it into the b64 image encoder function
                    # page_matrix = fz.Matrix(2, 2)
                    # page_pix = page.get_pixmap(matrix=page_matrix)
                    
                    # # retrieve the image bytes and pass it into the function
                    # page_image_bytes = page_pix.tobytes("png")
                    b64_encoded_page_image = render_page_to_b64(page)

                    processed_page_image_data = ProcessedPage(
                        page_number = page.number,
                        extraction_method = "vision",
                        raw_content = None,
                        image_b64_encoded = b64_encoded_page_image
                    )
                    processed_page_data_list.append(processed_page_image_data)

            # print(f"Number of pages flagged for text extraction: {len([page for page in text_extraction_flagged_page_count if page == True])}")
            # print(f"Number of pages flagged for vision extraction: {len([page for page in text_extraction_flagged_page_count if page == False])}")
            logger.info(f"Number of pages flagged for text extraction: {len([page for page in text_extraction_flagged_page_count if page == True])}")
            logger.info(f"Number of pages flagged for vision extraction: {len([page for page in text_extraction_flagged_page_count if page == False])}")

            processed_document =  ProcessedDocument(
                file_path = file_path,
                total_pages = document.page_count,
                pages = processed_page_data_list
            )
            return processed_document


    elif validated_file_tuple[1] == False:
        # print(f"Incompatible file uploaded.\n")
        logger.error(f"Incompatible file uploaded.\n")


def image_encoder(image_bytes: bytes) -> str:
    """Encode image bytes or scanned document page bytes to a base64 string.

    Args:
        image_bytes (bytes): Raw image bytes that need to be encoded.

    Returns:
        str: The resultant base64 string.
    """

    if not image_bytes:
        raise ImageEncodingException(
            message="Cannot encode empty image bytes"
        )

    try:
        return base64.b64encode(image_bytes).decode()
    except Exception as err:
        raise ImageEncodingException(
            message=f"Exception occurred while trying to encode the image bytes: {err}"
        ) from err

def extract_clinical_signals(processed_document: ProcessedDocument | ProcessedImage) -> dict:
    """Passes the extracted data into the Gemma models and returns a dict containing clinical signals/indicators

    Args:
        processed_document (ProcessedDocument | ProcessedImage): _description_

    Returns:
        dict: A dict object (intended to be JSON) containing clinical signals
    """

    # check whether the input is an image or document list and branch accordingly
    match processed_document:
        case ProcessedDocument():
            # here, each page needs to be looped over and parsed according to whether it needs an image extraction or text extraction
            # a single prompt is leveraged for this
            """prompt construction:
            * have the base system prompt that instructs the model on what it needs to perform
            * have a prompt holder -- a base str that will hold the entire prompt
            
            - if the page needs to a text extraction, extract text and add it to the prompt holder
            - if the page needs vision extraction, convert it to b64 encoding, and add it as an image with a placeholder [according to the gemma 4 prompt format]

            continue this step for all pages
            """

            # this will hold the instructions for the model
            base_system_prompt = """You are a Medical Document Analyst specialising in extracting clinical signals from medical documents (medical reports, lab results, pathology documents, radiology notes). You understand that an ordinary individual who does not have knowledge of understanding or interpreting needs more than just guidance; they need to be able to understand what they are looking at, signals that might have been overlooked, and the long term implications of the report they hold. And for that, you need to first extract the clinical signals from the document(s), which is what you do. You help in identifying clinical signals such as elevated markers, abnormal findings, flagged terms, and extracting them from the document. These signals are required to build a clinical profile of the patient.

            Extract key clinical signals from this medical document focusing on:
                1. Lab test results (normal vs abnormal ranges)
                2. Medication dosages (conversion units if needed)
                3. Imaging findings (shape, location, contrast)
                4. Patient demographics (age/gender/chat)
                5. Diagnosis implications

            Extract the clinical signals carefully with accuracy and precision. Construct a structured clinical profile of the patient using the extracted data, and provide the clinical profile as a single JSON object, following the provided JSON schema below strictly:
            
            {
                "document_type": "lab_report | radiology | pathology | clinical_note | unknown",
                "patient_context": {
                    "age": "number or null",
                    "sex": "string or null",
                    "stated_history": "string or null"
                },
                "lab_findings": [
                    {
                    "test_name": "string",
                    "value": "number or string",
                    "unit": "string or null",
                    "reference_range": "string or null",
                    "status": "normal | low | high | critical | unknown"
                    }
                ],
                "imaging_findings": [
                    {
                    "modality": "X-ray | MRI | CT | Ultrasound | other",
                    "region": "string",
                    "observation": "string"
                    }
                ],
                "clinical_notes": ["string"],
                "flagged_signals": [
                    {
                    "signal": "string",
                    "reason": "string"
                    }
                ],
                "extraction_confidence": "high | medium | low"
            }
            """ 

            content_parts = [] # this will hold the data from all the pages

            # attach the instructions for the model - the system prompt
            content_parts.append({
                "type": "text",
                "text": base_system_prompt
            })

            file_path = processed_document.file_path
            processed_document_pages = processed_document.pages

            # access each ProcessedPage object in the list
            for processed_page in processed_document_pages:
                # branch according to the extraction_method specified for the respective page
                match processed_page.extraction_method:
                    case "text":
                        content_parts.append({
                            "type": "text",
                            "text": processed_page.raw_content
                        })
                    case "vision":
                        content_parts.append({
                            "type": "image",
                            "image": processed_page.image_b64_encoded # handled by the processor
                        })
            

        case ProcessedImage():
            # this here is just an image uploaded by the user and we assume that there is no additional context provided by them
            if processed_document.image_b64_encoded is not None:
                # extract the b64 encodings
                b64_image_bytes = processed_document.image_b64_encoded
                # add it to the list of contents
                content_parts.append({
                    "type": "image",
                    "image": b64_image_bytes
                })
            else:
                logger.error(f"ProcessedImageObject does not contain b64 encodings. Cannot proceed with vision extraction")
                return

        # here, you need to pass the content_parts to the gemma4 model call (to be implemented, need more clarity on how)
        case _:
            return "unknown type"

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
            result = extraction_branching(validation_tuple)
            # logger.info(f"Result: {result}")

            # call the clinical signals extraction function
            if result is not None:
                clinical_signals = extract_clinical_signals(result)
                logger.info(f"Clinical signals: {clinical_signals}")
            print("-------\n")

if __name__ == "__main__":
    main()
