import io
import json
import random
import sys
import multiprocessing as mp
import tqdm
import istarmap

from PIL import Image, ImageDraw

from escriptorium_connector import EscriptoriumConnector
import os
from dotenv import load_dotenv
import requests
import numpy as np

load_dotenv('my.env')
escriptorium_url = str(os.getenv('ESCRIPTORIUM_URL'))
token = str(os.getenv('ESCRIPTORIUM_TOKEN'))
headers = {'Authorization': 'Token ' + token}


class Document:
    """
    Hack to re-compute polygons for first and last line of each given zone.
    """

    def __init__(self, document_pk, 
                 page_pk_boundaries: tuple, 
                 transcription_pk=None, 
                 main_zone_pk=None,
                 main_zone_label=None,
                 prefix=None,
                 document_name=None,
                 proportion=1):
        self.document_pk = document_pk
        self.region_types = None
        self.pages_list = range(page_pk_boundaries[0], page_pk_boundaries[1] + 1)
        self.interesting_lines = []
        self.page_pk = None
        self.correct_labels = main_zone_pk
        self.target_zone_pk_typology_pk = None
        self.transcriptions = None
        self.first_page_pk = page_pk_boundaries[0]
        self.last_page_pk = page_pk_boundaries[1]
        self.transcription_pk = transcription_pk
        self.target_zone_pk = main_zone_pk
        self.target_zone_label = main_zone_label
        self.chars_coords_dict = {}
        self.image_extension = None
        self.prefix = prefix
        self.document_name = document_name
        self.proportion = proportion

        self.escriptorium_url = str(os.getenv('ESCRIPTORIUM_URL'))
        username = str(os.getenv('ESCRIPTORIUM_USERNAME'))
        password = str(os.getenv('ESCRIPTORIUM_PASSWORD'))
        self.escr_connect = EscriptoriumConnector(self.escriptorium_url, username, password)
        self.get_regions_and_transcriptions()

    def get_page(self, page):
        url = f"{self.escriptorium_url}/api/documents/{self.document_pk}/parts/?page={page}"
        res = requests.get(url, headers=headers)
        try:
            data = res.json()
        except json.decoder.JSONDecodeError as e:
            print(res)
        else:
            for part in data['results']:
                self.pages_list.append(part['pk'])
            if data['next']:
                self.get_page(page + 1)

    def get_regions_and_transcriptions(self):
        document_base_json = requests.get(f'{self.escriptorium_url}/api/documents/{self.document_pk}/',
                                          headers=headers).json()
        print(document_base_json)

        self.region_types = {element['pk']: element['name'] for element in document_base_json['valid_block_types']}
        self.transcriptions = {transcr['pk']: transcr['name'] for transcr in document_base_json["transcriptions"]}
        print(f"Transcriptions pk: {self.transcriptions}")
        print(f"Region types: {self.region_types}\n\n")

    def get_lines_pk_from_region(self, page):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{page}'
        res = requests.get(parts_url, headers=headers).json()
        regions = res["regions"]
        dictionnary = {}
        simplified_regions = []
        if main_zone_pk:
            for region in regions:
                identifier = region['pk']
                typology = region['typology']
                try:
                    dictionnary[identifier] = self.region_types[typology]
                except KeyError as e:
                    print("Dict exception")
                    print(e)
                    print(region)
                    print(typology)
                    exit(0)
                if self.region_types[typology] == self.target_zone_pk:
                    self.target_zone_pk_typology_pk = typology
                    simplified_regions.append(identifier)
            id_order_typology_list = []

            for line in res['lines']:
                if dictionnary[line['region']] == self.target_zone_label:
                    id_order_typology_list.append(line['pk'])
        else:
            id_order_typology_list = []
            for line in res['lines']:
                id_order_typology_list.append(line['pk'])

        return id_order_typology_list

    def retrieve_chars_from_lines(self, part_pk, page_number, line_pk):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part_pk}/lines/{line_pk}'
        res = requests.get(parts_url, headers=headers).json()

        # On prend le premier, mais il faut filtrer sur le bon numéro.
        for index, transcriptions in enumerate(res["transcriptions"]):
            if transcriptions["transcription"] == self.transcription_pk:
                good_index = index

        transcriptions = res['transcriptions'][good_index]
        characters = transcriptions['graphs']
        for char in characters:
            random_integer = random.randint(0, 100)
            prop = int((1 - self.proportion) * 100)
            if random_integer > prop:
                pass
            else:
                continue
            # Déjà on s'assure que la clé existe
            try:
                self.chars_coords_dict[page_number]
            except:
                self.chars_coords_dict[page_number] = {}

            try:
                self.chars_coords_dict[page_number][char['c']].append(char['poly'])
            except Exception as e:
                self.chars_coords_dict[page_number][char['c']] = [char['poly']]

    def get_transcriptions_pk(self, part):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part}/transcriptions/'
        print(parts_url)
        res = requests.get(parts_url, headers=headers).json()
        print(res)
        with open("trash/test.json", "w") as json_output_file:
            json.dump(res, json_output_file)

    def get_image(self, page):
        url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{page}'
        page_as_json = requests.get(url, headers=headers).json()
        uri = page_as_json['image']['uri']
        res = requests.get(f"{self.escriptorium_url}/{uri}", headers=headers)
        img_name = uri.rsplit('/')[-1]
        self.image_extension = f".{img_name.split('.')[-1]}"
        try:
            os.mkdir(f"img/{self.document_name}")
        except:
            pass
        try:
            image = Image.open(io.BytesIO(res.content))
            image.save(f"img/{self.document_name}/{img_name}")
        except:
            print('no success for: ' + img_name)
        return img_name.replace(self.image_extension, "").replace(self.prefix, "")

    def open_image(self, page):
        image_path = f"img/{self.document_name}/{self.prefix}{page}{self.image_extension}"
        image = Image.open(image_path).convert("RGBA")
        return np.asarray(image)


def extract_images(page, coordonnees, char, index, image_as_array, document_name):
    """
    Extrait les images des lignes à partir des coordonnées extraites auparavant.
    :return: None
    """
    # https://stackoverflow.com/a/22650239



    # Il faut ajuster ici en fonction des caracteres. Creer des classes avec correction differente ?
    x_left_correction = - 18
    x_right_correction = + 25
    y_top_correction = + 10
    y_bottom_correction = - 10
    # https://stackoverflow.com/a/43591567
    # On selectionne le rectangle qui contient la ligne (= les valeurs d'abcisse et d'ordonnée
    # maximales et minimales)
    y_max = max([i[1] for i in coordonnees]) + y_bottom_correction
    y_min = min([i[1] for i in coordonnees]) + y_top_correction
    x_max = max([i[0] for i in coordonnees]) + x_right_correction
    x_min = min([i[0] for i in coordonnees]) + x_left_correction
    rectangle_coordinates = (x_min, y_min, x_max, y_max)
    polygone = ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    maskIm = Image.new('L', (image_as_array.shape[1], image_as_array.shape[0]), 0)  # c'est ici qu'on initialise
    # une plus petite image.
    ImageDraw.Draw(maskIm).polygon(polygone, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(image_as_array.shape, dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:, :, :3] = image_as_array[:, :, :3]

    # transparency (4th column)
    newImArray[:, :, 3] = mask * 255

    # On enregistre
    newIm = Image.fromarray(newImArray, "RGBA")
    try:
        cropped_img = newIm.crop(rectangle_coordinates)
    except:
        return
    try:
        os.mkdir(f"img/{document_name}/graphemes")
    except:
        pass
    try:
        os.mkdir(f"img/{document_name}/graphemes/{char}")
    except:
        pass
    try:
        cropped_img.save(f"img/{document_name}/graphemes/{char}/{page}_{index}.png")
    except:
        pass


if __name__ == '__main__':
    with open(sys.argv[1], "r") as conf_file:
        conf_dict = json.load(conf_file)

    document_pk = conf_dict["document_pk"]
    transcription_pk = conf_dict["transcription_pk"]
    main_zone_pk = conf_dict["target_zone_pk"]
    main_zone_label = conf_dict["target_zone_label"]
    file_prefix = conf_dict["prefix"]
    document_name = conf_dict["docName"]
    parts = conf_dict["parts"]
    proportion = conf_dict["proportion_to_keep"]

    document = Document(document_pk=document_pk,
                        document_name=document_name,
                        page_pk_boundaries=parts,
                        transcription_pk=transcription_pk,
                        main_zone_pk=main_zone_pk,
                        main_zone_label=main_zone_label,
                        prefix=file_prefix,
                        proportion=proportion)

    for page_pk in document.pages_list:
        page_number = document.get_image(page_pk)
        lines_list = document.get_lines_pk_from_region(page_pk)
        print(page_number)
        for line in lines_list:
            document.retrieve_chars_from_lines(page_pk, page_number, line)
        image_as_array = document.open_image(page_number)

        for char, realizations in document.chars_coords_dict[page_number].items():
            if char != " ":
                print(char)
                with mp.Pool(processes=int(2)) as pool:
                    data = [(page_number, coords, char, index, image_as_array, document.document_name) for index, coords
                            in enumerate(realizations)]
                    for _ in tqdm.tqdm(pool.istarmap(extract_images, data), total=len(data)):
                        pass
