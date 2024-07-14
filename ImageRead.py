import os
import time
from PIL import Image, ImageDraw
import requests, json

#pip install azure-cognitiveservices-vision-computervision==0.7.0

# import namespaces
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

def GetTextRead(image_file):
    # Get Configuration Settings
    cog_endpoint = "https://tanujfrenchreader.cognitiveservices.azure.com/"
    cog_key = "a9c9b1d7b0714cf2a2310b6c6b99cc50"

    # Authenticate Azure AI Vision client
    # Authenticate Azure AI Vision client
    credential = CognitiveServicesCredentials(cog_key)
    cv_client = ComputerVisionClient(cog_endpoint, credential)
    #print('Reading text in {}\n'.format(image_file))
    # Use Read API to read text in image
    with open(image_file, mode="rb") as image_data:
        read_op = cv_client.read_in_stream(image_data, raw=True)

        # Get the async operation ID so we can check for the results
        operation_location = read_op.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Wait for the asynchronous operation to complete
        while True:
            read_results = cv_client.get_read_result(operation_id)
            if read_results.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(1)

        # If the operation was successfully, process the text line by line
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
                    # Uncomment the following line if you'd like to see the bounding box
                    # print(line.bounding_box)
    return save