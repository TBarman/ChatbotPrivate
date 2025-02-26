import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import os

def GetTextRead(image_file):
    load_dotenv()
    cog_endpoint = os.getenv('COG_ENDPOINT')
    cog_key = os.getenv('COG_KEY')

    credential = CognitiveServicesCredentials(cog_key)
    cv_client = ComputerVisionClient(cog_endpoint, credential)

    with open(image_file, mode="rb") as image_data:
        read_op = cv_client.read_in_stream(image_data, raw=True)

        operation_location = read_op.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            read_results = cv_client.get_read_result(operation_id)
            if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)

        save = ""
        mod = 0
        if read_results.status == OperationStatusCodes.succeeded:
            for page in read_results.analyze_result.read_results:
                for line in page.lines:
                    if mod == 1:
                        mod = 0
                        save += line.text + "\n"
                    else:
                        mod = 1
                        save += line.text + ","

    return save
