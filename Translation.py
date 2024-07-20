import os
import requests
from dotenv import load_dotenv

def main():

    load_dotenv()
    global translator_endpoint
    global cog_key
    global cog_region

    try:
        cog_key = os.getenv('COG_KEY')
        cog_region = os.getenv('COG_REGION')
        translator_endpoint = os.getenv('TRANSLATOR_ENDPOINT')

        reviews_folder = 'reviews_folder'
        for file_name in os.listdir(reviews_folder):
            print('\n-------------\n' + file_name)
            text = open(os.path.join(reviews_folder, file_name), encoding='utf8').read()
            print('\n' + text)
            language = GetLanguage(text)
            print('Language:', language)
            if language != 'en':
                translation = Translate(text, language)
                print("\nTranslation:\n{}".format(translation))

    except Exception as ex:
        print(ex)


def GetLanguage(text):
    language = 'en'
    cog_key = os.getenv('COG_KEY')
    cog_region = os.getenv('COG_REGION')
    translator_endpoint = os.getenv('TRANSLATOR_ENDPOINT')
    
    path = '/detect'
    url = translator_endpoint + path

    # Build the request
    params = {
        'api-version': '3.0'
    }

    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json'
    }

    body = [{
        'text': text
    }]

    request = requests.post(url, params=params, headers=headers, json=body)
    response = request.json()
    language = response[0]["language"]

    return language


def Translate(text, source_language):
    cog_key = os.getenv('COG_KEY')
    cog_region = os.getenv('COG_REGION')
    translator_endpoint = os.getenv('TRANSLATOR_ENDPOINT')
    path = '/translate'
    url = translator_endpoint + path
    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': ['en']
    }

    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json'
    }

    body = [{
        'text': text
    }]

    request = requests.post(url, params=params, headers=headers, json=body)
    response = request.json()
    translation = response[0]["translations"][0]["text"]

    return translation


if __name__ == "__main__":
    main()
