import io
import json
import random
import shutil
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

# load_dotenv('params/example.env')
load_dotenv('params/my.env')
escriptorium_url = str(os.getenv('ESCRIPTORIUM_URL'))
token = str(os.getenv('ESCRIPTORIUM_TOKEN'))
headers = {'Authorization': 'Token ' + token}


class Document:
    """
    Classe document autour de laquelle tourne tout le script
    """

    def __init__(self,
                 document_pk,
                 page_pk_boundaries: tuple,
                 transcription_pk=None,
                 main_zone_pk=None,
                 main_zone_label=None,
                 prefix="",
                 document_name=None,
                 proportion=1,
                 graphemes_classification=None,
                 pixel_adjustments=None):
        """

        :param document_pk: identifiant de document
        :param page_pk_boundaries: identifiant de première et dernière page choisie
        :param transcription_pk: identifiant de transcription voulue
        :param main_zone_pk: (opt) identifiant de zone voulue
        :param main_zone_label: (opt) label de zone voulue
        :param prefix: (opt) préfixe du nom de chaque fichier s'il existe
        :param document_name: nom du document pour production du dossier de sortie
        :param proportion: proportion de graphèmes extraits
        :param graphemes_classification: classification des graphemes selon leur position part rapport à la ligne
        :param pixel_adjustments: paramètres d'ajustemenent par rapport aux classes précédentes
        """
        self.document_pk = document_pk
        self.pages_list = range(page_pk_boundaries[0], page_pk_boundaries[1] + 1)
        self.interesting_lines = []
        self.page_pk = None
        self.correct_labels = main_zone_pk
        self.target_zone_pk_typology_pk = None
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
        self.graphemes_classification = graphemes_classification
        self.pixel_adjustments = pixel_adjustments
        self.image_as_array = None
        self.page_name = None

        self.escriptorium_url = str(os.getenv('ESCRIPTORIUM_URL'))
        username = str(os.getenv('ESCRIPTORIUM_USERNAME'))
        password = str(os.getenv('ESCRIPTORIUM_PASSWORD'))
        self.escr_connect = EscriptoriumConnector(self.escriptorium_url, username, password)

        if len(sys.argv) > 2:
            if sys.argv[2] == "--identifiers":
                self.get_identifiers()

            if sys.argv[2] == "--classes":
                self.get_classes()

    def get_classes(self):
        """
        Cette fonction imprime tous les différents caractères présent dans les pages transcrites,
        et termine le programme
        :return: None
        """
        classes = set()
        for page_pk in tqdm.tqdm(self.pages_list):
            lines_list = self.get_lines_pk_from_region(page_pk)
            for line in lines_list:
                classes.update(self.get_different_chars_in_line(page_pk, line))

        classified_graphemes = set(list(self.graphemes_classification.keys()))
        difference = list(classes - classified_graphemes)
        assert len(difference) == 0, f"Please add {difference} to graphemes classification file."

        print(f"Found {len(classes)} classes: \n {list(classes)}")
        exit(0)

    def get_identifiers(self):
        """
        Cette fonction imprime les différents types de zone et de transcription ainsi que leur
        identifiant (pk)
        :return: None
        """
        url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/'
        document_base_json = requests.get(url, headers=headers).json()
        print(url)
        region_types = {element['pk']: element['name'] for element in document_base_json['valid_block_types']}
        transcriptions = {transcr['pk']: transcr['name'] for transcr in document_base_json["transcriptions"]}
        print(f"Transcriptions pk: {transcriptions}")
        print(f"Region types: {region_types}\n\n")
        exit(0)

    def get_lines_pk_from_region(self, page_pk: int) -> list:
        """
        Cette fonction retourne la liste de toutes les lignes d'une page donnée,
        en filtrant éventuellement par zone d'intérêt
        :param page:
        :return: La liste d'identifiants de ligne
        """
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{page_pk}'
        res = requests.get(parts_url, headers=headers).json()
        regions = res["regions"]
        dictionnary = {}
        simplified_regions = []
        list_of_lines_pk = []
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

            for line in res['lines']:
                if dictionnary[line['region']] == self.target_zone_label:
                    list_of_lines_pk.append(line['pk'])
        else:
            for line in res['lines']:
                list_of_lines_pk.append(line['pk'])

        return list_of_lines_pk

    def get_different_chars_in_line(self, part_pk, line_pk):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part_pk}/lines/{line_pk}'
        res = requests.get(parts_url, headers=headers).json()
        different_chars = []
        # On prend le premier, mais il faut filtrer sur le bon numéro.
        for index, transcriptions in enumerate(res["transcriptions"]):
            if transcriptions["transcription"] == self.transcription_pk:
                good_index = index

        transcriptions = res['transcriptions'][good_index]
        characters = transcriptions['graphs']

        return set([char['c'] for char in characters])


    def get_chars_from_lines(self, part_pk, line_pk):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part_pk}/lines/{line_pk}'
        res = requests.get(parts_url, headers=headers).json()

        # Il faut filtrer et récupérer la bonne transcription.
        for index, transcriptions in enumerate(res["transcriptions"]):
            if transcriptions["transcription"] == self.transcription_pk:
                correct_transcription_index = index

        try:
            transcriptions = res['transcriptions'][correct_transcription_index]
            characters = transcriptions['graphs']
            for char in characters:
                try:
                    os.makedirs(f"img/{self.document_name}/graphemes/{char['c']}")
                except:
                    pass
                random_integer = random.randint(0, 100)
                prop = int((1 - self.proportion) * 100)
                if random_integer > prop:
                    pass
                else:
                    continue
                # Déjà on s'assure que la clé existe
                try:
                    self.chars_coords_dict[self.page_name]
                except:
                    self.chars_coords_dict[self.page_name] = {}

                try:
                    self.chars_coords_dict[self.page_name][line_pk].append((char['c'], char['poly']))
                except Exception as e:
                    self.chars_coords_dict[self.page_name][line_pk] = [(char['c'], char['poly'])]
        except:
            pass




    def retrieve_chars_from_lines(self, part_pk, line_pk):
        """
        Cette fonction récupère les coordonnées de chaque réalisation d'un graphème et les ajoute à un dictionnaire
        de la forme:
        {"nom_de_la_page": {"grapheme": [coordonnées_realisation_1, coordonnées_realisation_2, ..., coordonnées_realisation_n]}}.
        Une fonction aléatoire vient permettre de ne conserver qu'une proportion donnée des réalisations si besoin.
        :param part_pk: l'identifiant de la page
        :param page_name: le nom réduit de la page
        :param line_pk: l'identifiant de la ligne
        :return: None
        """
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part_pk}/lines/{line_pk}'
        res = requests.get(parts_url, headers=headers).json()

        # Il faut filtrer et récupérer la bonne transcription.
        for index, transcriptions in enumerate(res["transcriptions"]):
            if transcriptions["transcription"] == self.transcription_pk:
                correct_transcription_index = index

        transcriptions = res['transcriptions'][correct_transcription_index]
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
                self.chars_coords_dict[self.page_name]
            except:
                self.chars_coords_dict[self.page_name] = {}

            try:
                self.chars_coords_dict[self.page_name][char['c']].append(char['poly'])
            except Exception as e:
                self.chars_coords_dict[self.page_name][char['c']] = [char['poly']]

    def get_transcriptions_pk(self, part):
        parts_url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{part}/transcriptions/'
        print(parts_url)
        res = requests.get(parts_url, headers=headers).json()
        print(res)
        with open("trash/test.json", "w") as json_output_file:
            json.dump(res, json_output_file)

    def get_image_name(self, page_pk):
        """
        Cette fonction télécharge l'image et renvoie le nom de l'image sans préfixe (s'il existe) ni extension:
        pg_0001.png > renvoie 0001.
        Fonction adaptée du vademecum sur l'API produit par Daniel Stökl et Robin Tissot:
        https://docs.google.com/document/d/1tl48eXHq36KJ1zyXq0dMwYEzdnQYUm_MKfzMat9vjPc/edit#
        :param page_pk: l'identifiant de la page
        :return:
        """
        url = f'{self.escriptorium_url}/api/documents/{self.document_pk}/parts/{page_pk}'
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
        self.page_name = img_name.replace(self.image_extension, "").replace(self.prefix, "")

    def open_image(self):
        """
        Ouvre une image et la convertit en matrice numpy.
        :param page_name: le nom de la page
        :return: la matrice numpy
        """
        image_path = f"img/{self.document_name}/{self.prefix}{self.page_name}{self.image_extension}"
        image = Image.open(image_path).convert("RGBA")
        self.image_as_array = np.asarray(image)


def extract_images(page,
                   line_pk,
                   chars,
                   chars_to_exclude,
                   only_chars,
                   image_as_array,
                   document_name,
                   pixel_adjustments: dict,
                   graphemes_classification: dict):
    """
    Extrait les images des graphemes à partir des coordonnées extraites auparavant,
    et les enregistre dans un dossier au nom du grapheme, de la forme:
    NomDeLaPage_NombreDeRealisationPrecedentes.extension


    :return: None
    """
    # https://stackoverflow.com/a/22650239

    # On a des classes avec des corrections différentes en fonction du grapheme
    if only_chars is None:
        pass
    else:
        chars = [(char, coords) for (char, coords) in chars if char in only_chars]
    for index, (char, coords) in enumerate(chars):
        if os.path.exists(f"img/{document_name}/graphemes/{char}/{page}_{line_pk}_{index}.png"):
            continue
        else:
            if char in chars_to_exclude:
                continue
            try:
                x_left_correction = pixel_adjustments[graphemes_classification[char]]["x_left_correction"]
            except KeyError:
                print(f"Char '{char}' not included in classification file, please do so.")
            x_right_correction = pixel_adjustments[graphemes_classification[char]]["x_right_correction"]
            y_top_correction = pixel_adjustments[graphemes_classification[char]]["y_top_correction"]
            y_bottom_correction = pixel_adjustments[graphemes_classification[char]]["y_bottom_correction"]

            # https://stackoverflow.com/a/43591567
            # On selectionne le rectangle qui contient la ligne (= les valeurs d'abcisse et d'ordonnée
            # maximales et minimales)
            y_max = max([i[1] for i in coords]) + y_bottom_correction
            y_min = min([i[1] for i in coords]) + y_top_correction
            x_max = max([i[0] for i in coords]) + x_right_correction
            x_min = min([i[0] for i in coords]) + x_left_correction

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
                cropped_img.save(f"img/{document_name}/graphemes/{char}/{page}_{line_pk}_{index}.png")
            except:
                pass


if __name__ == '__main__':
    with open(sys.argv[1], "r") as conf_file:
        conf_dict = json.load(conf_file)

    with open("params/graphemes_classification.json", "r") as conf_file:
        graphemes_classes = json.load(conf_file)

    if len(sys.argv) == 3:
        only_chars = sys.argv[2].split(',')
    else:
        only_chars = None

    # On modifie le dictionnaire pour le rendre plus efficace:
    graphemes_classification = {}
    for classe, graphemes in graphemes_classes.items():
        for grapheme in graphemes:
            graphemes_classification[grapheme] = classe

    document_pk = conf_dict["document_pk"]
    transcription_pk = conf_dict["transcription_pk"]
    main_zone_pk = conf_dict["target_zone_pk"]
    main_zone_label = conf_dict["target_zone_label"]
    file_prefix = conf_dict["prefix"]
    document_name = conf_dict["docName"]
    parts = conf_dict["parts"]
    proportion = conf_dict["proportion_to_keep"]
    chars_to_exclude = conf_dict["chars_to_exclude"]
    pixel_adjustments = conf_dict["ajustement"]

    document = Document(document_pk=document_pk,
                        document_name=document_name,
                        page_pk_boundaries=parts,
                        transcription_pk=transcription_pk,
                        main_zone_pk=main_zone_pk,
                        main_zone_label=main_zone_label,
                        prefix=file_prefix,
                        proportion=proportion,
                        graphemes_classification=graphemes_classification,
                        pixel_adjustments=pixel_adjustments)


    for page_pk in document.pages_list:
        document.get_image_name(page_pk)
        print(document.page_name)
        lines_list = document.get_lines_pk_from_region(page_pk)
        for line in lines_list:
            document.get_chars_from_lines(page_pk, line)
            # document.retrieve_chars_from_lines(page_pk, line)
        document.open_image()

        # On passe par du multiprocessing pour accélérer un peu les choses.
        with mp.Pool(processes=int(1)) as pool:
            data = [(document.page_name,
                     line_pk,
                     line,
                     chars_to_exclude,
                     only_chars,
                     document.image_as_array,
                     document.document_name,
                     document.pixel_adjustments,
                     document.graphemes_classification) for line_pk, line in
                    document.chars_coords_dict[document.page_name].items()]
            for _ in tqdm.tqdm(pool.istarmap(extract_images, data), total=len(data)):
                pass
